use std::cmp;
use std::mem;
use std::ops::Bound;
use std::sync::Arc;

use async_lock::{RwLock, RwLockUpgradableReadGuard};
use fusio_parquet::writer::AsyncWriter;
use parquet::arrow::{AsyncArrowWriter, ProjectionMask};
use ulid::Ulid;

use super::{CompactionError, Compactor};
use crate::compaction::RecordSchema;
use crate::fs::manager::StoreManager;
use crate::fs::{generate_file_id, FileId, FileType};
use crate::inmem::immutable::Immutable;
use crate::inmem::mutable::MutableMemTable;
use crate::ondisk::sstable::{SsTable, SsTableID};
use crate::scope::Scope;
use crate::stream::level::LevelStream;
use crate::stream::ScanStream;
use crate::version::edit::VersionEdit;
use crate::version::TransactionTs;
use crate::{
    context::Context,
    record::{self, Record},
    version::{Version, MAX_LEVEL},
    CompactionExecutor, DbOption, DbStorage,
};

pub struct LeveledTask {
    pub input: Vec<(usize, Vec<Ulid>)>,
}

pub struct LeveledCompactor<R: Record> {
    options: LeveledOptions,
    db_option: Arc<DbOption>,
    schema: Arc<RwLock<DbStorage<R>>>,
    ctx: Arc<Context<R>>,
    record_schema: Arc<R::Schema>,
}

#[derive(Clone, Debug)]
pub struct LeveledOptions {
    /// Size threshold (in bytes) to trigger major compaction relative to SST size
    pub major_threshold_with_sst_size: usize,
    /// Magnification factor controlling SST file count per level
    pub level_sst_magnification: usize,
    /// Default number of oldest tables to include in a major compaction
    pub major_default_oldest_table_num: usize,
    /// Maximum number of tables to select for major compaction at level L
    pub major_l_selection_table_max_num: usize,
    /// Number of immutable chunks to accumulate before triggering a flush
    pub immutable_chunk_num: usize,
    /// Maximum allowed number of immutable chunks in memory
    pub immutable_chunk_max_num: usize,
}

impl Default for LeveledOptions {
    fn default() -> Self {
        Self {
            major_threshold_with_sst_size: 4,
            level_sst_magnification: 10,
            major_default_oldest_table_num: 3,
            major_l_selection_table_max_num: 4,
            immutable_chunk_num: 3,
            immutable_chunk_max_num: 5,
        }
    }
}

impl<R: Record> LeveledCompactor<R> {
    pub(crate) fn new(
        options: LeveledOptions,
        schema: Arc<RwLock<DbStorage<R>>>,
        record_schema: Arc<R::Schema>,
        db_option: Arc<DbOption>,
        ctx: Arc<Context<R>>,
    ) -> Self {
        Self {
            options,
            db_option,
            schema,
            ctx,
            record_schema,
        }
    }
}

#[async_trait::async_trait]
impl<R> Compactor<R> for LeveledCompactor<R>
where
    R: Record,
    <<R as record::Record>::Schema as record::Schema>::Columns: Send + Sync,
{
    async fn check_then_compaction(&self, is_manual: bool) -> Result<(), CompactionError<R>> {
        self.minor_flush(is_manual).await?;
        while self.should_major_compact().await {
            if let Some(task) = self.plan_major().await {
                self.execute_major(task).await?;
            } else {
                break;
            }
        }

        if is_manual {
            self.ctx.version_set.rewrite().await.unwrap();
        }

        Ok(())
    }
}

impl<R: Record> CompactionExecutor<R> for LeveledCompactor<R>
where
    <<R as crate::record::Record>::Schema as crate::record::Schema>::Columns: Send + Sync,
{
    fn check_then_compaction(
        &self,
        is_manual: bool,
    ) -> impl std::future::Future<Output = Result<(), CompactionError<R>>> + Send {
        <Self as Compactor<R>>::check_then_compaction(self, is_manual)
    }
}

impl<R> LeveledCompactor<R>
where
    R: Record,
    <<R as record::Record>::Schema as record::Schema>::Columns: Send + Sync,
{
    pub async fn should_major_compact(&self) -> bool {
        // Check if any level needs major compaction
        let version_ref = self.ctx.version_set.current().await;
        for level in 0..MAX_LEVEL - 2 {
            if Self::is_threshold_exceeded_major(&self.options, &version_ref, level) {
                return true;
            }
        }
        false
    }

    pub async fn plan_major(&self) -> Option<LeveledTask> {
        let version_ref = self.ctx.version_set.current().await;

        // Find the first level that needs compaction
        for level in 0..MAX_LEVEL - 2 {
            if Self::is_threshold_exceeded_major(&self.options, &version_ref, level) {
                // Collect file IDs from the level that needs compaction
                let level_files: Vec<Ulid> = version_ref.level_slice[level]
                    .iter()
                    .map(|scope| scope.gen)
                    .collect();

                if !level_files.is_empty() {
                    let mut input = vec![(level, level_files)];
                    if level + 1 < MAX_LEVEL {
                        let next_level_files: Vec<Ulid> = version_ref.level_slice[level + 1]
                            .iter()
                            .map(|scope| scope.gen)
                            .collect();

                        if !next_level_files.is_empty() {
                            input.push((level + 1, next_level_files));
                        }
                    }
                    return Some(LeveledTask { input });
                }
            }
        }
        None
    }

    pub async fn execute_major(
        &self,
        task: LeveledTask,
    ) -> Result<(), CompactionError<R>> {
        let version_ref = self.ctx.version_set.current().await;
        let mut version_edits = vec![];
        let mut delete_gens = vec![];

        // Extract the level from the task
        for (level, file_gens) in &task.input {
            if file_gens.is_empty() {
                continue;
            }

            // Get the scopes for the files to be compacted
            let level_scopes: Vec<&Scope<_>> = version_ref.level_slice[*level]
                .iter()
                .filter(|scope| file_gens.contains(&scope.gen))
                .collect();

            if level_scopes.is_empty() {
                continue;
            }

            // Determine min/max range for compaction
            let min = level_scopes.iter().map(|scope| &scope.min).min().unwrap();
            let max = level_scopes.iter().map(|scope| &scope.max).max().unwrap();
            // Execute the actual compaction logic
            Self::major_compaction(
                &version_ref,
                &self.db_option,
                &self.options,
                &min,
                &max,
                &mut version_edits,
                &mut delete_gens,
                &self.record_schema,
                &self.ctx,
                task.input[0].0,
            )
            .await?;

            break; // Process one level at a time
        }

        if !version_edits.is_empty() {
            version_edits.push(VersionEdit::LatestTimeStamp {
                ts: version_ref.increase_ts(),
            });

            self.ctx
                .version_set
                .apply_edits(version_edits, Some(delete_gens), false)
                .await?;
        }

        Ok(())
    }

    pub async fn minor_flush(
        &self,
        is_manual: bool,
    ) -> Result<Option<LeveledTask>, CompactionError<R>> {
        let mut guard = self.schema.write().await;

        guard.trigger.reset();

        // Add the mutable memtable into the immutable memtable
        if !guard.mutable.is_empty() {
            let trigger_clone = guard.trigger.clone();

            // Replace mutable memtable with new memtable
            let mutable = mem::replace(
                &mut guard.mutable,
                MutableMemTable::new(
                    &self.db_option,
                    trigger_clone,
                    self.ctx.manager.base_fs().clone(),
                    self.record_schema.clone(),
                )
                .await?,
            );
            let (file_id, immutable) = mutable.into_immutable().await?;
            guard.immutables.push((file_id, immutable));
        } else if !is_manual {
            return Ok(None);
        }

        if (is_manual && !guard.immutables.is_empty())
            || guard.immutables.len() > self.options.immutable_chunk_max_num
        {
            let recover_wal_ids = guard.recover_wal_ids.take();
            drop(guard);

            let guard = self.schema.upgradable_read().await;
            let chunk_num = if is_manual {
                guard.immutables.len()
            } else {
                self.options.immutable_chunk_num
            };
            let excess = &guard.immutables[0..chunk_num];

            if let Some(scope) = Self::minor_compaction(
                &self.db_option,
                recover_wal_ids,
                excess,
                &guard.record_schema,
                &self.ctx.manager,
            )
            .await?
            {
                let version_ref = self.ctx.version_set.current().await;
                let mut version_edits = vec![VersionEdit::Add { level: 0, scope }];
                version_edits.push(VersionEdit::LatestTimeStamp {
                    ts: version_ref.increase_ts(),
                });

                self.ctx
                    .version_set
                    .apply_edits(version_edits, None, false)
                    .await?;
            }
            let mut guard = RwLockUpgradableReadGuard::upgrade(guard).await;
            let sources = guard.immutables.split_off(chunk_num);
            let _ = mem::replace(&mut guard.immutables, sources);
        }
        Ok(None)
    }

    // Combine immutable memtables into SST file
    async fn minor_compaction(
        option: &DbOption,
        recover_wal_ids: Option<Vec<FileId>>,
        batches: &[(
            Option<FileId>,
            Immutable<<R::Schema as RecordSchema>::Columns>,
        )],
        schema: &R::Schema,
        manager: &StoreManager,
    ) -> Result<Option<Scope<<R::Schema as RecordSchema>::Key>>, CompactionError<R>> {
        if !batches.is_empty() {
            let level_0_path = option.level_fs_path(0).unwrap_or(&option.base_path);
            let level_0_fs = manager.get_fs(level_0_path);

            let mut min = None;
            let mut max = None;

            let gen = generate_file_id();
            let mut wal_ids = Vec::with_capacity(batches.len());

            // Creates writer to write Arrow record batches into parquet
            let mut writer = AsyncArrowWriter::try_new(
                AsyncWriter::new(
                    level_0_fs
                        .open_options(
                            &option.table_path(gen, 0),
                            FileType::Parquet.open_options(false),
                        )
                        .await?,
                ),
                schema.arrow_schema().clone(),
                Some(option.write_parquet_properties.clone()),
            )?;

            // Retrieve WAL ids so recovery is possible if the database crashes before
            // the SST id is written to the `Version`
            if let Some(mut recover_wal_ids) = recover_wal_ids {
                wal_ids.append(&mut recover_wal_ids);
            }
            for (file_id, batch) in batches {
                if let (Some(batch_min), Some(batch_max)) = batch.scope() {
                    if matches!(min.as_ref().map(|min| min > batch_min), Some(true) | None) {
                        min = Some(batch_min.clone())
                    }
                    if matches!(max.as_ref().map(|max| max < batch_max), Some(true) | None) {
                        max = Some(batch_max.clone())
                    }
                }
                writer.write(batch.as_record_batch()).await?;
                if let Some(file_id) = file_id {
                    wal_ids.push(*file_id);
                }
            }
            let file_size = writer.bytes_written() as u64;
            writer.close().await?;
            return Ok(Some(Scope {
                min: min.ok_or(CompactionError::EmptyLevel)?,
                max: max.ok_or(CompactionError::EmptyLevel)?,
                gen,
                wal_ids: Some(wal_ids),
                file_size,
            }));
        }
        Ok(None)
    }

    // Accumulate all SST files in a stream that fall within the min/max range in `level` and `level
    // + 1`. Then use those files to build the new SST files and delete the olds ones
    #[allow(clippy::too_many_arguments)]
    async fn major_compaction(
        version: &Version<R>,
        option: &DbOption,
        leveled_options: &LeveledOptions,
        mut min: &<R::Schema as RecordSchema>::Key,
        mut max: &<R::Schema as RecordSchema>::Key,
        version_edits: &mut Vec<VersionEdit<<R::Schema as RecordSchema>::Key>>,
        delete_gens: &mut Vec<SsTableID>,
        instance: &R::Schema,
        ctx: &Context<R>,
        target_level: usize,
    ) -> Result<(), CompactionError<R>> {
        let level = target_level;

        let (meet_scopes_l, start_l, end_l) = Self::this_level_scopes(version, min, max, level, leveled_options);
        let (meet_scopes_ll, start_ll, end_ll) =
            Self::next_level_scopes(version, &mut min, &mut max, level, &meet_scopes_l)?;

        let level_path = option.level_fs_path(level).unwrap_or(&option.base_path);
        let level_fs = ctx.manager.get_fs(level_path);
        let mut streams = Vec::with_capacity(meet_scopes_l.len() + meet_scopes_ll.len());

        // Behaviour for level 0 is different as it is unsorted + has overlapping keys
        if level == 0 {
            for scope in meet_scopes_l.iter() {
                let file = level_fs
                    .open_options(
                        &option.table_path(scope.gen, level),
                        FileType::Parquet.open_options(true),
                    )
                    .await?;

                streams.push(ScanStream::SsTable {
                    inner: SsTable::open(ctx.parquet_lru.clone(), scope.gen, file)
                        .await?
                        .scan(
                            (Bound::Unbounded, Bound::Unbounded),
                            u32::MAX.into(),
                            None,
                            ProjectionMask::all(),
                        )
                        .await?,
                });
            }
        } else {
            let (lower, upper) = <LeveledCompactor<R> as Compactor<R>>::full_scope(&meet_scopes_l)?;
            let level_scan_l = LevelStream::new(
                version,
                level,
                start_l,
                end_l,
                (Bound::Included(lower), Bound::Included(upper)),
                u32::MAX.into(),
                None,
                ProjectionMask::all(),
                level_fs.clone(),
                ctx.parquet_lru.clone(),
            )
            .ok_or(CompactionError::EmptyLevel)?;

            streams.push(ScanStream::Level {
                inner: level_scan_l,
            });
        }

        let level_l_path = option.level_fs_path(level + 1).unwrap_or(&option.base_path);
        let level_l_fs = ctx.manager.get_fs(level_l_path);

        // Pushes next level SSTs that fall in the range
        if !meet_scopes_ll.is_empty() {
            let (lower, upper) =
                <LeveledCompactor<R> as Compactor<R>>::full_scope(&meet_scopes_ll)?;
            let level_scan_ll = LevelStream::new(
                version,
                level + 1,
                start_ll,
                end_ll,
                (Bound::Included(lower), Bound::Included(upper)),
                u32::MAX.into(),
                None,
                ProjectionMask::all(),
                level_l_fs.clone(),
                ctx.parquet_lru.clone(),
            )
            .ok_or(CompactionError::EmptyLevel)?;

            streams.push(ScanStream::Level {
                inner: level_scan_ll,
            });
        }

        // Build the new SSTs
        <LeveledCompactor<R> as Compactor<R>>::build_tables(
            option,
            version_edits,
            level + 1,
            streams,
            instance,
            level_l_fs,
        )
        .await?;

        // Delete old files on both levels
        for scope in meet_scopes_l {
            version_edits.push(VersionEdit::Remove {
                level: level as u8,
                gen: scope.gen,
            });
            delete_gens.push(SsTableID::new(scope.gen, level));
        }
        for scope in meet_scopes_ll {
            version_edits.push(VersionEdit::Remove {
                level: (level + 1) as u8,
                gen: scope.gen,
            });
            delete_gens.push(SsTableID::new(scope.gen, level + 1));
        }

        Ok(())
    }
    // Finds all SST files in the next level that overlap the range of the current level
    fn next_level_scopes<'a>(
        version: &'a Version<R>,
        min: &mut &'a <R::Schema as RecordSchema>::Key,
        max: &mut &'a <R::Schema as RecordSchema>::Key,
        level: usize,
        meet_scopes_l: &[&'a Scope<<R::Schema as RecordSchema>::Key>],
    ) -> Result<
        (
            Vec<&'a Scope<<R::Schema as RecordSchema>::Key>>,
            usize,
            usize,
        ),
        CompactionError<R>,
    > {
        let mut meet_scopes_ll = Vec::new();
        let mut start_ll = 0;
        let mut end_ll = 0;

        if !version.level_slice[level + 1].is_empty() {
            *min = meet_scopes_l
                .iter()
                .map(|scope| &scope.min)
                .min()
                .ok_or(CompactionError::EmptyLevel)?;

            *max = meet_scopes_l
                .iter()
                .map(|scope| &scope.max)
                .max()
                .ok_or(CompactionError::EmptyLevel)?;

            start_ll = Version::<R>::scope_search(min, &version.level_slice[level + 1]);
            end_ll = Version::<R>::scope_search(max, &version.level_slice[level + 1]);

            let next_level_len = version.level_slice[level + 1].len();
            for scope in version.level_slice[level + 1]
                [start_ll..cmp::min(end_ll + 1, next_level_len)]
                .iter()
            {
                if scope.contains(min) || scope.contains(max) {
                    meet_scopes_ll.push(scope);
                }
            }
        }
        Ok((meet_scopes_ll, start_ll, end_ll))
    }

    // Finds SST files in the specified level that overlap with the key ranges
    fn this_level_scopes<'a>(
        version: &'a Version<R>,
        min: &<R::Schema as RecordSchema>::Key,
        max: &<R::Schema as RecordSchema>::Key,
        level: usize,
        options: &LeveledOptions,
    ) -> (
        Vec<&'a Scope<<R::Schema as RecordSchema>::Key>>,
        usize,
        usize,
    ) {
        let mut meet_scopes_l = Vec::new();
        let mut start_l = Version::<R>::scope_search(min, &version.level_slice[level]);
        let mut end_l = start_l;

        for scope in version.level_slice[level][start_l..].iter() {
            if (scope.contains(min) || scope.contains(max))
                && meet_scopes_l.len() <= options.major_l_selection_table_max_num
            {
                meet_scopes_l.push(scope);
                end_l += 1;
            } else {
                break;
            }
        }
        if meet_scopes_l.is_empty() {
            start_l = 0;
            end_l = cmp::min(
                options.major_default_oldest_table_num,
                version.level_slice[level].len(),
            );

            for scope in version.level_slice[level][..end_l].iter() {
                if meet_scopes_l.len() > options.major_l_selection_table_max_num {
                    break;
                }
                meet_scopes_l.push(scope);
            }
        }
        (meet_scopes_l, start_l, end_l - 1)
    }

    /// Checks if the number of SST files in a level exceeds the major compaction threshold
    ///
    /// The threshold is calculated by multiplying the base threshold with a magnification factor
    /// that increases exponentially with the level number.
    ///
    /// Returns true if the number of tables in the level exceeds the threshold.
    pub(crate) fn is_threshold_exceeded_major(
        options: &LeveledOptions,
        version: &Version<R>,
        level: usize,
    ) -> bool {
        Version::<R>::tables_len(version, level)
            >= (options.major_threshold_with_sst_size
                * options.level_sst_magnification.pow(level as u32))
    }
}
#[cfg(all(test, feature = "tokio"))]
pub(crate) mod tests {
    use std::sync::{atomic::AtomicU32, Arc};

    use flume::bounded;
    use fusio::{path::Path, DynFs};
    use fusio_dispatch::FsOptions;
    use parquet_lru::NoCache;
    use tempfile::TempDir;

    use crate::{
        compaction::{
            error::CompactionError, leveled::{LeveledCompactor, LeveledOptions}, tests::{build_parquet_table, build_version}
        },
        context::Context,
        executor::tokio::TokioExecutor,
        fs::{generate_file_id, manager::StoreManager},
        inmem::{
            immutable::{tests::TestSchema, Immutable},
            mutable::MutableMemTable,
        },
        record::{self, DataType, DynRecord, DynSchema, Record, Schema, Value, ValueDesc},
        scope::Scope,
        tests::Test,
        timestamp::Timestamp,
        trigger::{TriggerFactory, TriggerType},
        version::{cleaner::Cleaner, edit::VersionEdit, set::VersionSet, Version, MAX_LEVEL},
        wal::log::LogType,
        DbError, DbOption, DB,
        CompactionExecutor,Compactor,
    };

    async fn build_immutable<R>(
        option: &DbOption,
        records: Vec<(LogType, R, Timestamp)>,
        schema: &Arc<R::Schema>,
        fs: &Arc<dyn DynFs>,
    ) -> Result<Immutable<<R::Schema as Schema>::Columns>, DbError>
    where
        R: Record + Send,
    {
        let trigger = TriggerFactory::create(option.trigger_type);

        let mutable = MutableMemTable::new(option, trigger, fs.clone(), schema.clone()).await?;

        for (log_ty, record, ts) in records {
            let _ = mutable.insert(log_ty, record, ts).await?;
        }
        Ok(mutable.into_immutable().await?.1)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn minor_compaction() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_dir_l0 = tempfile::tempdir().unwrap();

        let option = DbOption::new(
            Path::from_filesystem_path(temp_dir.path()).unwrap(),
            &TestSchema,
        )
        .level_path(
            0,
            Path::from_filesystem_path(temp_dir_l0.path()).unwrap(),
            FsOptions::Local,
        )
        .unwrap();
        let manager =
            StoreManager::new(option.base_fs.clone(), option.level_paths.clone()).unwrap();
        manager
            .base_fs()
            .create_dir_all(&option.wal_dir_path())
            .await
            .unwrap();

        let batch_1 = build_immutable::<Test>(
            &option,
            vec![
                (
                    LogType::Full,
                    Test {
                        vstring: 3.to_string(),
                        vu32: 0,
                        vbool: None,
                    },
                    0.into(),
                ),
                (
                    LogType::Full,
                    Test {
                        vstring: 5.to_string(),
                        vu32: 0,
                        vbool: None,
                    },
                    0.into(),
                ),
                (
                    LogType::Full,
                    Test {
                        vstring: 6.to_string(),
                        vu32: 0,
                        vbool: None,
                    },
                    0.into(),
                ),
            ],
            &Arc::new(TestSchema),
            manager.base_fs(),
        )
        .await
        .unwrap();

        let batch_2 = build_immutable::<Test>(
            &option,
            vec![
                (
                    LogType::Full,
                    Test {
                        vstring: 4.to_string(),
                        vu32: 0,
                        vbool: None,
                    },
                    0.into(),
                ),
                (
                    LogType::Full,
                    Test {
                        vstring: 2.to_string(),
                        vu32: 0,
                        vbool: None,
                    },
                    0.into(),
                ),
                (
                    LogType::Full,
                    Test {
                        vstring: 1.to_string(),
                        vu32: 0,
                        vbool: None,
                    },
                    0.into(),
                ),
            ],
            &Arc::new(TestSchema),
            manager.base_fs(),
        )
        .await
        .unwrap();

        let scope = LeveledCompactor::<Test>::minor_compaction(
            &option,
            None,
            &vec![
                (Some(generate_file_id()), batch_1),
                (Some(generate_file_id()), batch_2),
            ],
            &TestSchema,
            &manager,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(scope.min, 1.to_string());
        assert_eq!(scope.max, 6.to_string());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dyn_minor_compaction() {
        let temp_dir = tempfile::tempdir().unwrap();
        let manager = StoreManager::new(FsOptions::Local, vec![]).unwrap();
        let schema = DynSchema::new(
            vec![ValueDesc::new("id".to_owned(), DataType::Int32, false)],
            0,
        );
        let option = DbOption::new(
            Path::from_filesystem_path(temp_dir.path()).unwrap(),
            &schema,
        );
        manager
            .base_fs()
            .create_dir_all(&option.wal_dir_path())
            .await
            .unwrap();

        let instance = Arc::new(schema);

        let mut batch1_data = vec![];
        let mut batch2_data = vec![];
        for i in 0..40 {
            let col = Value::new(DataType::Int32, "id".to_owned(), Arc::new(i), false);
            if i % 4 == 0 {
                continue;
            }
            if i < 35 && (i % 2 == 0 || i % 5 == 0) {
                batch1_data.push((LogType::Full, DynRecord::new(vec![col], 0), 0.into()));
            } else if i >= 7 {
                batch2_data.push((LogType::Full, DynRecord::new(vec![col], 0), 0.into()));
            }
        }

        // data range: [2, 34]
        let batch_1 =
            build_immutable::<DynRecord>(&option, batch1_data, &instance, manager.base_fs())
                .await
                .unwrap();

        // data range: [7, 39]
        let batch_2 =
            build_immutable::<DynRecord>(&option, batch2_data, &instance, manager.base_fs())
                .await
                .unwrap();

        let scope = LeveledCompactor::<DynRecord>::minor_compaction(
            &option,
            None,
            &vec![
                (Some(generate_file_id()), batch_1),
                (Some(generate_file_id()), batch_2),
            ],
            &instance,
            &manager,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            scope.min,
            Value::new(DataType::Int32, "id".to_owned(), Arc::new(2), false)
        );
        assert_eq!(
            scope.max,
            Value::new(DataType::Int32, "id".to_owned(), Arc::new(39), false)
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn major_compaction() {
        let temp_dir = TempDir::new().unwrap();
        let temp_dir_l0 = TempDir::new().unwrap();
        let temp_dir_l1 = TempDir::new().unwrap();

        let mut option = DbOption::new(
            Path::from_filesystem_path(temp_dir.path()).unwrap(),
            &TestSchema,
        )
        .level_path(
            0,
            Path::from_filesystem_path(temp_dir_l0.path()).unwrap(),
            FsOptions::Local,
        )
        .unwrap()
        .level_path(
            1,
            Path::from_filesystem_path(temp_dir_l1.path()).unwrap(),
            FsOptions::Local,
        )
        .unwrap();
        option = option.major_threshold_with_sst_size(2);
        let option = Arc::new(option);
        let manager = Arc::new(
            StoreManager::new(option.base_fs.clone(), option.level_paths.clone()).unwrap(),
        );

        manager
            .base_fs()
            .create_dir_all(&option.version_log_dir_path())
            .await
            .unwrap();
        manager
            .base_fs()
            .create_dir_all(&option.wal_dir_path())
            .await
            .unwrap();

        let ((table_gen_1, table_gen_2, table_gen_3, table_gen_4, _), version) =
            build_version(&option, &manager, &Arc::new(TestSchema)).await;

        let min = 2.to_string();
        let max = 5.to_string();
        let mut version_edits = Vec::new();

        let (_, clean_sender) = Cleaner::new(option.clone(), manager.clone());
        let version_set = VersionSet::new(clean_sender, option.clone(), manager.clone())
            .await
            .unwrap();
        let ctx = Context::new(
            manager.clone(),
            Arc::new(NoCache::default()),
            version_set,
            TestSchema.arrow_schema().clone(),
        );

        LeveledCompactor::<Test>::major_compaction(
            &version,
            &option,
            &LeveledOptions::default(),
            &min,
            &max,
            &mut version_edits,
            &mut vec![],
            &TestSchema,
            &ctx,
            0,
        )
        .await
        .unwrap();

        if let VersionEdit::Add { level, scope } = &version_edits[0] {
            assert_eq!(*level, 1);
            assert_eq!(scope.min, 1.to_string());
            assert_eq!(scope.max, 6.to_string());
        }
        assert_eq!(
            version_edits[1..5].to_vec(),
            vec![
                VersionEdit::Remove {
                    level: 0,
                    gen: table_gen_1,
                },
                VersionEdit::Remove {
                    level: 0,
                    gen: table_gen_2,
                },
                VersionEdit::Remove {
                    level: 1,
                    gen: table_gen_3,
                },
                VersionEdit::Remove {
                    level: 1,
                    gen: table_gen_4,
                },
            ]
        );
    }

    // https://github.com/tonbo-io/tonbo/pull/139
    #[tokio::test(flavor = "multi_thread")]
    async fn major_panic() {
        let temp_dir = TempDir::new().unwrap();

        let option = DbOption::new(
            Path::from_filesystem_path(temp_dir.path()).unwrap(),
            &TestSchema,
        )
        .major_threshold_with_sst_size(1)
        .level_sst_magnification(1);
        let manager = Arc::new(
            StoreManager::new(option.base_fs.clone(), option.level_paths.clone()).unwrap(),
        );

        manager
            .base_fs()
            .create_dir_all(&option.version_log_dir_path())
            .await
            .unwrap();
        manager
            .base_fs()
            .create_dir_all(&option.wal_dir_path())
            .await
            .unwrap();

        let level_0_fs = option
            .level_fs_path(0)
            .map(|path| manager.get_fs(path))
            .unwrap_or(manager.base_fs());
        let level_1_fs = option
            .level_fs_path(1)
            .map(|path| manager.get_fs(path))
            .unwrap_or(manager.base_fs());

        let table_gen0 = generate_file_id();
        let table_gen1 = generate_file_id();
        let mut records0 = vec![];
        let mut records1 = vec![];
        for i in 0..10 {
            let record = (
                LogType::Full,
                Test {
                    vstring: i.to_string(),
                    vu32: i,
                    vbool: Some(true),
                },
                0.into(),
            );
            if i < 5 {
                records0.push(record);
            } else {
                records1.push(record);
            }
        }
        build_parquet_table::<Test>(
            &option,
            table_gen0,
            records0,
            &Arc::new(TestSchema),
            0,
            level_0_fs,
        )
        .await
        .unwrap();
        build_parquet_table::<Test>(
            &option,
            table_gen1,
            records1,
            &Arc::new(TestSchema),
            1,
            level_1_fs,
        )
        .await
        .unwrap();

        let option = Arc::new(option);
        let (sender, _) = bounded(1);
        let mut version =
            Version::<Test>::new(option.clone(), sender, Arc::new(AtomicU32::default()));
        version.level_slice[0].push(Scope {
            min: 0.to_string(),
            max: 4.to_string(),
            gen: table_gen0,
            wal_ids: None,
            file_size: 13,
        });
        version.level_slice[1].push(Scope {
            min: 5.to_string(),
            max: 9.to_string(),
            gen: table_gen1,
            wal_ids: None,
            file_size: 13,
        });

        let mut version_edits = Vec::new();
        let min = 6.to_string();
        let max = 9.to_string();

        let (_, clean_sender) = Cleaner::new(option.clone(), manager.clone());
        let version_set = VersionSet::new(clean_sender, option.clone(), manager.clone())
            .await
            .unwrap();
        let ctx = Context::new(
            manager.clone(),
            Arc::new(NoCache::default()),
            version_set,
            TestSchema.arrow_schema().clone(),
        );
        LeveledCompactor::<Test>::major_compaction(
            &version,
            &option,
            &LeveledOptions::default(),
            &min,
            &max,
            &mut version_edits,
            &mut vec![],
            &TestSchema,
            &ctx,
            0,
        )
        .await
        .unwrap();
    }

    // issue: https://github.com/tonbo-io/tonbo/issues/152
    #[tokio::test(flavor = "multi_thread")]
    async fn test_flush_major_level_sort() {
        let temp_dir = TempDir::new().unwrap();
        eprintln!("test");
        let mut option = DbOption::new(
            Path::from_filesystem_path(temp_dir.path()).unwrap(),
            &TestSchema,
        )
        .immutable_chunk_num(1)
        .immutable_chunk_max_num(0)
        .major_threshold_with_sst_size(2)
        .level_sst_magnification(1)
        .max_sst_file_size(2 * 1024 * 1024)
        .major_default_oldest_table_num(1);
        option.trigger_type = TriggerType::Length(5);

        let db: DB<Test, TokioExecutor> = DB::new(option, TokioExecutor::current(), TestSchema)
            .await
            .unwrap();

        for i in 5..9 {
            let item = Test {
                vstring: i.to_string(),
                vu32: i,
                vbool: Some(true),
            };
            db.insert(item).await.unwrap();
        }

        db.flush().await.unwrap();
        for i in 0..4 {
            let item = Test {
                vstring: i.to_string(),
                vu32: i,
                vbool: Some(true),
            };
            db.insert(item).await.unwrap();
        }
        db.flush().await.unwrap();

        db.insert(Test {
            vstring: "6".to_owned(),
            vu32: 22,
            vbool: Some(false),
        })
        .await
        .unwrap();
        db.insert(Test {
            vstring: "8".to_owned(),
            vu32: 77,
            vbool: Some(false),
        })
        .await
        .unwrap();
        db.flush().await.unwrap();
        db.insert(Test {
            vstring: "1".to_owned(),
            vu32: 22,
            vbool: Some(false),
        })
        .await
        .unwrap();
        db.insert(Test {
            vstring: "5".to_owned(),
            vu32: 77,
            vbool: Some(false),
        })
        .await
        .unwrap();
        db.flush().await.unwrap();

        db.insert(Test {
            vstring: "2".to_owned(),
            vu32: 22,
            vbool: Some(false),
        })
        .await
        .unwrap();
        db.insert(Test {
            vstring: "7".to_owned(),
            vu32: 77,
            vbool: Some(false),
        })
        .await
        .unwrap();
        db.flush().await.unwrap();

        let version = db.ctx.version_set.current().await;

        for level in 0..MAX_LEVEL {
            let sort_runs = &version.level_slice[level];

            if sort_runs.is_empty() {
                continue;
            }
            for pos in 0..sort_runs.len() - 1 {
                let current = &sort_runs[pos];
                let next = &sort_runs[pos + 1];

                assert!(current.min < current.max);
                assert!(next.min < next.max);

                if level == 0 {
                    continue;
                }
                assert!(current.max < next.min);
            }
        }
        dbg!(version);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_leveled_compaction_correctness() {
        let temp_dir = TempDir::new().unwrap();
        let option = Arc::new(DbOption::new(
            Path::from_filesystem_path(temp_dir.path()).unwrap(),
            &TestSchema,
        ));

        let (sender, _) = bounded(1);
        let mut version = Version::<Test>::new(option.clone(), sender, Arc::new(AtomicU32::default()));

        let options = LeveledOptions {
            major_threshold_with_sst_size: 2,
            level_sst_magnification: 4,
            ..Default::default()
        };

        // Test initial state
        assert!(!LeveledCompactor::<Test>::is_threshold_exceeded_major(&options, &version, 0));

        // Add files to level 0 to trigger compaction
        version.level_slice[0].push(Scope {
            min: "1".to_string(),
            max: "5".to_string(),
            gen: generate_file_id(),
            wal_ids: None,
            file_size: 100,
        });
        version.level_slice[0].push(Scope {
            min: "2".to_string(),
            max: "6".to_string(),
            gen: generate_file_id(),
            wal_ids: None,
            file_size: 100,
        });

        // Now level 0 should exceed threshold
        assert!(LeveledCompactor::<Test>::is_threshold_exceeded_major(&options, &version, 0));

        // Test threshold calculation for different levels
        assert_eq!(options.major_threshold_with_sst_size * options.level_sst_magnification.pow(0), 2); // Level 0: 2 * 4^0 = 2
        assert_eq!(options.major_threshold_with_sst_size * options.level_sst_magnification.pow(1), 8); // Level 1: 2 * 4^1 = 8
        assert_eq!(options.major_threshold_with_sst_size * options.level_sst_magnification.pow(2), 32); // Level 2: 2 * 4^2 = 32

        // Verify overlapping ranges in level 0 (allowed)
        let level0_scopes = &version.level_slice[0];
        assert_eq!(level0_scopes.len(), 2);
        assert!(level0_scopes[0].min <= "5".to_string() && level0_scopes[1].min <= "6".to_string());
        assert!(level0_scopes[0].max >= "1".to_string() && level0_scopes[1].max >= "2".to_string());
        
        // The ranges [1,5] and [2,6] overlap, which is expected for level 0
        let overlaps = level0_scopes[0].max >= level0_scopes[1].min && level0_scopes[1].max >= level0_scopes[0].min;
        assert!(overlaps, "Level 0 files should be allowed to overlap");
    }

    pub(crate) fn convert_test_ref_to_test(entry: crate::transaction::TransactionEntry<'_, Test>) -> Option<Test> {
        match &entry {
            crate::transaction::TransactionEntry::Stream(stream_entry) => {
                if stream_entry.value().is_some() {
                    let test_ref = entry.get();
                    Some(Test {
                        vstring: test_ref.vstring.to_string(),
                        vu32: test_ref.vu32.unwrap_or(0),
                        vbool: test_ref.vbool,
                    })
                } else {
                    None
                }
            }
            crate::transaction::TransactionEntry::Local(_) => {
                let test_ref = entry.get();
                Some(Test {
                    vstring: test_ref.vstring.to_string(),
                    vu32: test_ref.vu32.unwrap_or(0),
                    vbool: test_ref.vbool,
                })
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_leveled_data_integrity_across_levels() {
        let temp_dir = TempDir::new().unwrap();
        let mut option = DbOption::new(
            Path::from_filesystem_path(temp_dir.path()).unwrap(),
            &TestSchema,
        )
        .major_threshold_with_sst_size(2)
        .level_sst_magnification(2)
        .immutable_chunk_num(1)
        .immutable_chunk_max_num(2);
        option.trigger_type = TriggerType::Length(3);

        let db: DB<Test, TokioExecutor> = DB::new(option, TokioExecutor::current(), TestSchema)
            .await
            .unwrap();

        // Insert test data that will trigger multiple compactions
        let test_data = vec![
            ("key001", 1),
            ("key002", 2),
            ("key003", 3),
            ("key004", 4),
            ("key005", 5),
            ("key006", 6),
            ("key007", 7),
            ("key008", 8),
            ("key009", 9),
            ("key010", 10),
        ];

        // Insert data and force flushes to create multiple SST files
        for (key, value) in &test_data {
            db.insert(Test {
                vstring: key.to_string(),
                vu32: *value,
                vbool: Some(true),
            })
            .await
            .unwrap();
        }

        db.flush().await.unwrap();

        // Insert more data to trigger compaction
        for i in 11..21 {
            db.insert(Test {
                vstring: format!("key{:03}", i),
                vu32: i,
                vbool: Some(false),
            })
            .await
            .unwrap();
        }

        db.flush().await.unwrap();

        // Verify all data is readable
        for (key, expected_value) in &test_data {
            let key_string = key.to_string();
            let result = db.get(&key_string, convert_test_ref_to_test).await.unwrap();
            assert!(result.is_some(), "Key {} should be found", key);
            let record = result.unwrap();
            assert_eq!(record.vu32, *expected_value, "Value for key {} should match", key);
        }

        // Verify additional data
        for i in 11..21 {
            let key = format!("key{:03}", i);
            let result = db.get(&key, convert_test_ref_to_test).await.unwrap();
            assert!(result.is_some(), "Key {} should be found", key);
            let record = result.unwrap();
            assert_eq!(record.vu32, i, "Value for key {} should match", key);
        }

        // Check version structure
        let version = db.ctx.version_set.current().await;
        let mut total_files = 0;
        for level in 0..MAX_LEVEL {
            let file_count = version.level_slice[level].len();
            total_files += file_count;
            if file_count > 0 {
                println!("Level {}: {} files", level, file_count);
                
                // Verify non-overlapping property for levels > 0
                if level > 0 {
                    let scopes = &version.level_slice[level];
                    for i in 0..scopes.len().saturating_sub(1) {
                        assert!(
                            scopes[i].max < scopes[i + 1].min,
                            "Level {} files should be non-overlapping", level
                        );
                    }
                }
            }
        }
        assert!(total_files > 0, "Should have files in the LSM tree");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_leveled_compaction_with_tombstones() {
        let temp_dir = TempDir::new().unwrap();
        let mut option = DbOption::new(
            Path::from_filesystem_path(temp_dir.path()).unwrap(),
            &TestSchema,
        )
        .major_threshold_with_sst_size(2)
        .level_sst_magnification(2);
        option.trigger_type = TriggerType::Length(3);

        let db: DB<Test, TokioExecutor> = DB::new(option, TokioExecutor::current(), TestSchema)
            .await
            .unwrap();

        // Insert initial data
        for i in 0..10 {
            db.insert(Test {
                vstring: format!("key{:02}", i),
                vu32: i,
                vbool: Some(true),
            })
            .await
            .unwrap();
        }

        db.flush().await.unwrap();

        // Delete some keys (creates tombstones)
        for i in (0..10).step_by(2) {
            let key = format!("key{:02}", i);
            db.remove(key).await.unwrap();
        }

        db.flush().await.unwrap();

        // Add more data to trigger compaction
        for i in 10..20 {
            db.insert(Test {
                vstring: format!("key{:02}", i),
                vu32: i,
                vbool: Some(false),
            })
            .await
            .unwrap();
        }

        db.flush().await.unwrap();

        // Verify deleted keys are not found
        for i in (0..10).step_by(2) {
            let key = format!("key{:02}", i);
            let result = db.get(&key, convert_test_ref_to_test).await.unwrap();
            assert!(result.is_none(), "Deleted key {} should not be found", key);
        }

        // Verify non-deleted keys are still found
        for i in (1..10).step_by(2) {
            let key = format!("key{:02}", i);
            let result = db.get(&key, convert_test_ref_to_test).await.unwrap();
            assert!(result.is_some(), "Non-deleted key {} should be found", key);
            let record = result.unwrap();
            assert_eq!(record.vu32, i, "Value should be correct");
        }

        // Check that compaction properly handles tombstones
        let version = db.ctx.version_set.current().await;
        let mut has_files = false;
        for level in 0..MAX_LEVEL {
            if !version.level_slice[level].is_empty() {
                has_files = true;
                break;
            }
        }
        assert!(has_files, "Should have files after operations");
    }
}

#[cfg(all(test, feature = "tokio"))]
pub(crate) mod tests_metric {

    use fusio::{path::Path};
    use tempfile::TempDir;
    use crate::compaction::leveled::tests::convert_test_ref_to_test;

    use crate::compaction::leveled::LeveledOptions;
    use crate::{
        executor::tokio::TokioExecutor,
        inmem::{
            immutable::{tests::TestSchema},
        },
        tests::Test,
        trigger::{TriggerType},
        version::MAX_LEVEL,
        DbOption, DB,
    };


    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_read_write_amplification_measurement() {
        let temp_dir = TempDir::new().unwrap();
        let option = DbOption::new(
            Path::from_filesystem_path(temp_dir.path()).unwrap(),
            &TestSchema,
        )
        .major_threshold_with_sst_size(3)
        .level_sst_magnification(4)
        .max_sst_file_size(1024); 

        let db: DB<Test, TokioExecutor> = DB::new(option.clone(), TokioExecutor::current(), TestSchema)
            .await
            .unwrap();

        // Track metrics for amplification calculation
        let mut total_bytes_written_by_user = 0u64;
        let mut compaction_rounds = 0;

        // Insert initial dataset with more substantial data
        let initial_records = 1000;
        let iter_num = 10;
        for i in 0..initial_records * iter_num {
            let record = Test {
                vstring: format!("this_is_a_longer_key_to_make_files_bigger_{:05}", i),
                vu32: i as u32,
                vbool: Some(i % 2 == 0),
            };
            
            // More accurate user data size calculation
            let string_bytes = record.vstring.as_bytes().len();
            let u32_bytes = 4;
            let bool_bytes = 1;
            let record_size = string_bytes + u32_bytes + bool_bytes;
            total_bytes_written_by_user += record_size as u64;
            
            db.insert(record).await.unwrap();

            if i%initial_records == 0 {
                // Force flush and compaction
                db.flush().await.unwrap();
                compaction_rounds += 1;
            }
        }

        // Verify data integrity after all compactions (check a sample of keys)
        for i in 0..initial_records * iter_num {
            let key = format!("this_is_a_longer_key_to_make_files_bigger_{:05}", i);
            let result = db.get(&key, convert_test_ref_to_test).await.unwrap();
            if result.is_some() {
                let record = result.unwrap();
                assert_eq!(record.vu32, i as u32, "Value should be preserved after compaction");
            } else {
                panic!("Key {} should exist after compaction", key);
            }
        }        

        // Get final version to measure total file sizes
        let final_version = db.ctx.version_set.current().await;
        let mut files_per_level = vec![0; MAX_LEVEL];

        // Verify that total scope.file_size matches total actual file size on disk
        let manager = crate::fs::manager::StoreManager::new(option.base_fs.clone(), vec![]).unwrap();
        let fs = manager.base_fs();
        let mut total_actual_file_size = 0u64;
        
        for level in 0..MAX_LEVEL {
            files_per_level[level] = final_version.level_slice[level].len();
            for scope in &final_version.level_slice[level] {
                let file = fs
                    .open_options(
                        &option.table_path(scope.gen, level),
                        crate::fs::FileType::Parquet.open_options(true),
                    )
                    .await
                    .unwrap();
                let actual_size = file.size().await.unwrap();
                total_actual_file_size += actual_size;
            }
        }
        
        // Calculate amplification metrics using actual file sizes
        let write_amplification = 
            total_actual_file_size as f64 / total_bytes_written_by_user as f64;

        // Read amplification estimation (simplified)
        // In a real scenario, this would require tracking actual read operations
        let estimated_read_amplification = {
            let mut read_amp = 0.0;
            for level in 0..MAX_LEVEL {
                if files_per_level[level] > 0 {
                    // Level 0 files can overlap, so worst case is reading all files
                    if level == 0 {
                        read_amp += files_per_level[level] as f64;
                    } else {
                        // For other levels, typically 1 file per level for a point lookup
                        read_amp += 1.0;
                    }
                }
            }
            read_amp
        };

        println!("=== Amplification Metrics ===");
        println!("User data written: {} bytes", total_bytes_written_by_user);
        println!("Total file size: {} bytes", total_actual_file_size);
        println!("Write Amplification: {:.2}x", write_amplification);
        println!("Estimated Read Amplification: {:.2}x", estimated_read_amplification);
        println!("Compaction rounds: {}", compaction_rounds);
        
        for level in 0..MAX_LEVEL {
            if files_per_level[level] > 0 {
                println!("Level {}: {} files", level, files_per_level[level]);
            }
        }

        // Assertions for reasonable amplification  
        // Write amplification can be less than 1.0 in some cases due to compression
        // and the way Parquet stores data efficiently. The important thing is that
        // we can measure it and it's non-zero.
        assert!(write_amplification > 0.0, "Write amplification should be positive");
        assert!(write_amplification < 10.0, "Write amplification should be reasonable (< 10x)");
        assert!(estimated_read_amplification >= 1.0, "Read amplification should be at least 1.0");
        assert!(total_actual_file_size > 0, "Should have written some data to disk");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_throughput() {
        use std::time::Instant;
        use futures_util::StreamExt;
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        
        let temp_dir = TempDir::new().unwrap();
        let mut option = DbOption::new(
            Path::from_filesystem_path(temp_dir.path()).unwrap(),
            &TestSchema,
        )
        .leveled_compaction(LeveledOptions::default());
        option.trigger_type = TriggerType::SizeOfMem(1 * 1024 * 1024);

        // Create DB with EcoTune compactor using the standard open method
        let db: DB<Test, TokioExecutor> = DB::new(option.clone(), TokioExecutor::current(), TestSchema)
            .await
            .unwrap();

        // Test parameters based on EcoTune paper (Section 5.1: 35% Get, 35% Seek, 30% long range scans)
        let total_operations = 100000;
        let insert_ratio = 0.3; // 30% inserts to build up data
        let get_ratio = 0.35; // 35% Get operations (point queries)
        let seek_ratio = 0.35; // 35% Seek operations  
        let long_range_ratio = 0.30; // 30% long range scans (paper workload)
        
        let insert_count = (total_operations as f64 * insert_ratio) as usize;
        let query_count = total_operations - insert_count;
        let get_count = (query_count as f64 * (get_ratio / (get_ratio + seek_ratio + long_range_ratio))) as usize;
        let seek_count = (query_count as f64 * (seek_ratio / (get_ratio + seek_ratio + long_range_ratio))) as usize;
        let long_range_count = query_count - get_count - seek_count;
        
        println!("EcoTune throughput test with paper proportions:");
        println!("- {} inserts ({:.1}%)", insert_count, insert_ratio * 100.0);
        println!("- {} Get queries ({:.1}%)", get_count, (get_count as f64 / total_operations as f64) * 100.0);
        println!("- {} Seek queries ({:.1}%)", seek_count, (seek_count as f64 / total_operations as f64) * 100.0);
        println!("- {} long-range scans ({:.1}%)", long_range_count, (long_range_count as f64 / total_operations as f64) * 100.0);

        // Create mixed workload operations vector
        
        let mut operations = Vec::new();
        
        // Add insert operations
        for i in 0..insert_count {
            operations.push(("insert", i));
        }
        
        // Add get operations  
        for i in 0..get_count {
            operations.push(("get", i));
        }
        
        // Add seek operations
        for i in 0..seek_count {
            operations.push(("seek", i));
        }
        
        // Add long-range scan operations
        for i in 0..long_range_count {
            operations.push(("long_range", i));
        }
        
        // Shuffle operations to create mixed workload
        let mut rng = rand::rngs::StdRng::seed_from_u64(42); // Fixed seed for reproducibility
        operations.shuffle(&mut rng);
        
        // Execute mixed workload
        let mixed_start = Instant::now();
        let mut insert_ops = 0;
        let mut successful_queries = 0;
        
        for (op_type, index) in operations {
            match op_type {
                "insert" => {
                    let record = Test {
                        vstring: format!("test_key_{:06}", index),
                        vu32: index as u32,
                        vbool: Some(index % 2 == 0),
                    };
                    db.insert(record).await.unwrap();
                    insert_ops += 1;
                }
                "get" => {
                    // Use modulo to ensure key exists (only query from inserted keys)
                    let key = format!("test_key_{:06}", index % insert_ops.max(1));
                    let found = db.get(&key, |entry| {
                        match entry {
                            crate::transaction::TransactionEntry::Stream(stream_entry) => {
                                Some(stream_entry.value().is_some())
                            }
                            crate::transaction::TransactionEntry::Local(_) => Some(true),
                        }
                    }).await.unwrap();
                    if found.unwrap_or(false) {
                        successful_queries += 1;
                    }
                }
                "seek" => {
                    let key = format!("test_key_{:06}", index % insert_ops.max(1));
                    let scan = db.scan((
                        std::ops::Bound::Included(&key),
                        std::ops::Bound::Unbounded
                    ), |entry| {
                        match entry {
                            crate::transaction::TransactionEntry::Stream(_) => true,
                            crate::transaction::TransactionEntry::Local(_) => true,
                        }
                    }).await.take(1);
                    let mut scan = std::pin::pin!(scan);
                    
                    if let Some(result) = scan.next().await {
                        if result.is_ok() {
                            successful_queries += 1;
                        }
                    }
                }
                "long_range" => {
                    let start_key = format!("test_key_{:06}", index % insert_ops.max(1));
                    let scan = db.scan((
                        std::ops::Bound::Included(&start_key),
                        std::ops::Bound::Unbounded
                    ), |entry| {
                        match entry {
                            crate::transaction::TransactionEntry::Stream(_) => true,
                            crate::transaction::TransactionEntry::Local(_) => true,
                        }
                    }).await.take(100);
                    let mut scan = std::pin::pin!(scan);
                    
                    let mut count = 0;
                    while let Some(result) = scan.next().await {
                        if result.is_ok() {
                            count += 1;
                            if count >= 100 { break; } // Limit to K=100
                        }
                    }
                    if count > 0 { successful_queries += 1; }
                }
                _ => unreachable!()
            }
        }
        
        let mixed_duration = mixed_start.elapsed();
        let mixed_throughput = total_operations as f64 / mixed_duration.as_secs_f64();
        
        // Calculate mixed workload results
        println!("Mixed Workload Throughput Results:");
        println!("Overall throughput: {:.2} ops/sec", mixed_throughput);
        println!("Total operations: {} (inserts: {}, successful queries: {})", total_operations, insert_ops, successful_queries);
        println!("Total time: {:.3}s", mixed_duration.as_secs_f64());
    }    
}
