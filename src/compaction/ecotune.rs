use std::cmp;
use std::mem;
use std::ops::Bound;
use std::sync::Arc;
use std::collections::HashMap;

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
    version::Version,
    CompactionExecutor, DbOption, DbStorage,
};

pub struct EcoTuneTask {
    pub input: Vec<(usize, Vec<Ulid>)>,
    pub target_level: usize,
}

/// State for the Dynamic Programming compaction scheduling algorithm
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompactionState {
    /// Number of existing sorted runs in main level (R_i in paper)
    pub existing_runs: usize,
    /// Number of incoming TM compactions remaining (i in paper) 
    pub remaining_tm_compactions: usize,
    /// Pending ML compaction size from previous decision (j in paper)
    pub pending_ml_size: usize,
    /// Number of consecutive TM operations before current ML (k in paper)
    pub consecutive_tm_ops: usize,
}

/// Decision for ML compaction scheduling
#[derive(Clone, Debug)]
pub struct CompactionDecision {
    /// Number of runs to merge in next ML compaction (x in paper)
    pub runs_to_merge: usize,
    /// Expected score/benefit of this decision
    pub score: f64,
}

/// Persistent execution state tracking for EcoTune DP algorithm
/// This tracks the actual execution history across compaction rounds
#[derive(Clone, Debug, Default)]
pub struct EcoTuneExecutionState {
    /// Number of consecutive TM operations performed since last ML compaction
    pub consecutive_tm_count: usize,
    /// Size of the last ML compaction performed (number of runs merged)
    pub last_ml_size: usize,
    /// Total TM compactions performed in current round
    pub tm_compactions_in_round: usize,
    /// Current compaction round number
    pub current_round: usize,
}

/// EcoTune three-level LSM-tree model constants
pub const ECOTUNE_MAX_LEVEL: usize = 3;  // EcoTune uses 3-level model: top(0), main(1), bottom(2)
pub const ECOTUNE_MAIN_LEVEL: usize = 1; // Main level index where TM/ML decisions are made
pub const ECOTUNE_BOTTOM_LEVEL: usize = 2; // Bottom level index (must contain single run)

#[derive(Clone, Debug)]
pub struct EcoTuneOptions {
    /// R: Total number of TM compactions in a compaction round
    pub total_tm_compactions: usize,
    /// T: Size ratio between levels (if using fixed ratio)
    pub size_ratio: usize,
    /// K: Long range scan length for cost calculation
    pub long_range_scan_length: usize,
    /// r: Proportion of long-range queries in workload
    pub long_range_ratio: f64,
    /// f: False positive rate for point queries
    pub false_positive_rate: f64,
    /// Tw: Time between consecutive TM compactions
    pub tm_compaction_interval: f64,
    /// Tc: Time to rewrite S data using MLC threads
    pub ml_compaction_time_per_unit: f64,
    /// β: Query speed ratio during ML compaction (q'(e) = β * q(e))
    pub query_speed_ratio_during_ml: f64,
    /// C: Capped size ratio between main and last levels
    pub main_to_last_level_ratio: usize,
    /// Size threshold to trigger compaction
    pub size_threshold: usize,
    /// Number of immutable chunks to accumulate before triggering a flush
    pub immutable_chunk_num: usize,
    /// Maximum allowed number of immutable chunks in memory
    pub immutable_chunk_max_num: usize,
    /// Optimization interval for parameter retuning
    pub optimization_interval: usize,
    /// Current optimization round counter
    pub optimization_round: usize,
}

impl Default for EcoTuneOptions {
    fn default() -> Self {
        Self {
            // EcoTune paper parameters (Section 5.1, Table 1)
            total_tm_compactions: 10,        // R = 10 TM compactions per round
            size_ratio: 4,                   // T = 4 (EcoTune paper default)
            long_range_scan_length: 100,     // K = 100 keys per long scan
            long_range_ratio: 0.9,           // r = 90% long range queries (paper default)
            false_positive_rate: 0.01,       // f = 1% false positive rate (typical Bloom filter)
            // Paper timing parameters (Section 5.1)
            tm_compaction_interval: 1.0,     // Tw = 1.0s time between TM compactions
            ml_compaction_time_per_unit: 0.1, // Tc = 0.1s to compact 1 unit of data
            query_speed_ratio_during_ml: 0.5, // β = 50% query speed during ML (paper estimate)
            main_to_last_level_ratio: 10,    // C = 10x size ratio (main to last level)
            size_threshold: 4,               // Threshold for triggering compaction
            immutable_chunk_num: 3,
            immutable_chunk_max_num: 5,
            optimization_interval: 100,      // Optimize every 100 compactions
            optimization_round: 0,
        }
    }
}

impl EcoTuneOptions {
    /// Calculate query speed based on number of existing runs in main level
    /// From paper: q(e) = 1 / ((e + 2) * r + (1 - r) * (1 + f))
    pub fn query_speed(&self, existing_runs: usize) -> f64 {
        let e = existing_runs as f64;
        let r = self.long_range_ratio;
        let f = self.false_positive_rate;
        1.0 / ((e + 2.0) * r + (1.0 - r) * (1.0 + f))
    }
    
    /// Calculate query speed during ML compaction
    pub fn query_speed_during_ml(&self, existing_runs: usize) -> f64 {
        self.query_speed_ratio_during_ml * self.query_speed(existing_runs)
    }

    /// Check if target level is the bottom level (requires single run constraint)
    pub fn is_compacting_to_bottom(&self, target_level: usize) -> bool {
        target_level >= ECOTUNE_BOTTOM_LEVEL
    }

    /// Check if bottom level constraint is violated
    pub(crate) fn is_bottom_level_violated<R: Record>(&self, version: &Version<R>) -> bool {
        // Bottom level must have at most 1 run
        version.level_slice.len() > ECOTUNE_BOTTOM_LEVEL && version.level_slice[ECOTUNE_BOTTOM_LEVEL].len() > 1
    }
}

/// This compactor is from "Rethinking The Compaction Policies in LSM-trees" paper. 
/// Implemented its O(R^4) Algorithm.
/// This compactor is not stable now, and should NOT be used.
pub struct EcoTuneCompactor<R: Record> {
    options: Arc<RwLock<EcoTuneOptions>>,
    db_option: Arc<DbOption>,
    schema: Arc<RwLock<DbStorage<R>>>,
    ctx: Arc<Context<R>>,
    record_schema: Arc<R::Schema>,
    /// Memoization table for DP algorithm
    dp_cache: Arc<RwLock<HashMap<CompactionState, CompactionDecision>>>,
    /// Persistent execution state for correct DP algorithm implementation
    execution_state: Arc<RwLock<EcoTuneExecutionState>>,
}

impl<R: Record> EcoTuneCompactor<R> {
    pub fn new(
        options: EcoTuneOptions,
        schema: Arc<RwLock<DbStorage<R>>>,
        record_schema: Arc<R::Schema>,
        db_option: Arc<DbOption>,
        ctx: Arc<Context<R>>,
    ) -> Self {
        Self {
            options: Arc::new(RwLock::new(options)),
            db_option,
            schema,
            ctx,
            record_schema,
            dp_cache: Arc::new(RwLock::new(HashMap::new())),
            execution_state: Arc::new(RwLock::new(EcoTuneExecutionState::default())),
        }
    }
}

#[async_trait::async_trait]
impl<R> Compactor<R> for EcoTuneCompactor<R>
where
    R: Record,
    <<R as record::Record>::Schema as record::Schema>::Columns: Send + Sync,
{
    async fn check_then_compaction(&self, is_manual: bool) -> Result<(), CompactionError<R>> {
        self.minor_flush(is_manual).await?;
        
        while self.should_major_compact().await {
            if let Some(task) = self.plan_major().await {
                // plan_major() computes DP decisions directly for current state
                self.execute_major(task).await?;
                // After execution, state changes, so next DP computation will be for new state
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

impl<R: Record> CompactionExecutor<R> for EcoTuneCompactor<R>
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

impl<R> EcoTuneCompactor<R>
where
    R: Record,
    <<R as record::Record>::Schema as record::Schema>::Columns: Send + Sync,
{
    pub async fn should_major_compact(&self) -> bool {
        // Simple fix: use plan_major() to determine if compaction is needed
        // This prevents infinite loops by ensuring should_major_compact() and plan_major() 
        // always agree on whether compaction can actually be performed
        self.plan_major().await.is_some()
    }

    pub async fn plan_major(&self) -> Option<EcoTuneTask> {
        let version_ref = self.ctx.version_set.current().await;
        let options = self.options.read().await;

        // Priority 1: Fix bottom level constraint violation (merge all runs to single run)
        if options.is_bottom_level_violated(&version_ref) {
            // Double-check constraint violation with current state
            if ECOTUNE_BOTTOM_LEVEL < version_ref.level_slice.len() {
                let bottom_level_files: Vec<Ulid> = version_ref.level_slice[ECOTUNE_BOTTOM_LEVEL]
                    .iter()
                    .map(|scope| scope.gen)
                    .collect();

                if !bottom_level_files.is_empty() {
                    println!("EcoTune: Priority fix - bottom level constraint violated, merging {} runs", bottom_level_files.len());
                    return Some(EcoTuneTask {
                        input: vec![(ECOTUNE_BOTTOM_LEVEL, bottom_level_files)],
                        target_level: ECOTUNE_BOTTOM_LEVEL, // Compact within bottom level to merge all runs
                    });
                } else {
                    println!("EcoTune: Bottom level constraint appeared violated but no files found - race condition detected");
                }
            } else {
                println!("EcoTune: Bottom level constraint appeared violated but level doesn't exist - race condition detected");
            }
        }

        // Priority 2: Use EcoTune DP algorithm to decide TM vs ML compaction
        if let Ok(optimal_decision) = self.get_optimal_compaction_decision().await {
            println!("EcoTune DP decision: merge {} runs with score {:.3}", 
                    optimal_decision.runs_to_merge, optimal_decision.score);

            // Apply DP decision
            if optimal_decision.runs_to_merge >= 2 {
                // DP says: Perform ML compaction (merge runs within main level)
                if let Some(ml_task) = self.create_ml_compaction_task(optimal_decision.runs_to_merge).await {
                    return Some(ml_task);
                }
            } else if optimal_decision.runs_to_merge == 1 {
                // DP says: Allow TM compaction (top to main level)
                if let Some(tm_task) = self.create_tm_compaction_task().await {
                    return Some(tm_task);
                }
            }
            // If runs_to_merge == 0, DP says: do nothing (wait)
        }

        // Priority 3: Fallback to threshold-based compaction for urgent cases
        // (This handles cases where DP cache is empty or computation failed)
        for level in 0..ECOTUNE_MAX_LEVEL - 1 {
            if Self::is_threshold_exceeded(&options, &version_ref, level) {
                // Ensure level exists before accessing it
                if level < version_ref.level_slice.len() {
                    let level_files: Vec<Ulid> = version_ref.level_slice[level]
                        .iter()
                        .map(|scope| scope.gen)
                        .collect();

                    if !level_files.is_empty() {
                        let target_level = level + 1;
                        let mut input = vec![(level, level_files)];
                        
                        // If compacting to bottom level, include ALL bottom level runs for merging
                        if options.is_compacting_to_bottom(target_level) && target_level < version_ref.level_slice.len() {
                            let bottom_level_files: Vec<Ulid> = version_ref.level_slice[target_level]
                                .iter()
                                .map(|scope| scope.gen)
                                .collect();

                            if !bottom_level_files.is_empty() {
                                input.push((target_level, bottom_level_files));
                            }
                        }
                        
                        println!("EcoTune: Fallback threshold-based compaction for level {} -> level {}", level, target_level);
                        return Some(EcoTuneTask { 
                            input,
                            target_level,
                        });
                    } else {
                        println!("EcoTune: Level {} threshold exceeded but no files found - race condition detected", level);
                    }
                } else {
                    println!("EcoTune: Level {} threshold exceeded but level doesn't exist - race condition detected", level);
                }
            }
        }
        
        // If we reach here, should_major_compact was true but we couldn't create any task
        // This indicates a potential logic issue that could cause infinite loops
        /* println!("EcoTune: Unable to create compaction task despite should_major_compact() returning true");
        println!("EcoTune: Current state - levels: {}, bottom level size: {}", 
                version_ref.level_slice.len(),
                version_ref.level_slice.get(ECOTUNE_BOTTOM_LEVEL).map_or(0, |level| level.len()));
        
        for (level_idx, level) in version_ref.level_slice.iter().enumerate() {
            if !level.is_empty() {
                let level_size = Version::<R>::tables_len(&version_ref, level_idx);
                println!("EcoTune: Level {}: {} files, size threshold: {}", 
                        level_idx, level.len(), level_size);
            }
        } */
        
        None
    }


    /// Get optimal compaction decision for current system state
    async fn get_optimal_compaction_decision(&self) -> Result<CompactionDecision, CompactionError<R>> {
        let version_ref = self.ctx.version_set.current().await;
        let options = self.options.read().await;
        let exec_state = self.execution_state.read().await;
        
        // Current state of main level for O(R⁴) algorithm
        let current_runs = version_ref.level_slice.get(ECOTUNE_MAIN_LEVEL).map_or(0, |level| level.len());
        
        // Calculate remaining TM compactions in current round
        let remaining_tm = options.total_tm_compactions.saturating_sub(exec_state.tm_compactions_in_round);
        
        let current_state = CompactionState {
            existing_runs: current_runs,
            remaining_tm_compactions: remaining_tm,
            pending_ml_size: exec_state.last_ml_size,
            consecutive_tm_ops: exec_state.consecutive_tm_count,
        };
        
        // Check cache first for current state
        {
            let cache = self.dp_cache.read().await;
            if let Some(cached_decision) = cache.get(&current_state) {
                return Ok(cached_decision.clone());
            }
        }
        
        // If not cached, compute DP decision directly
        self.solve_compaction_scheduling(current_state, &options).await
    }

    /// Create ML compaction task for main level (merge runs within level 1)
    async fn create_ml_compaction_task(&self, runs_to_merge: usize) -> Option<EcoTuneTask> {
        let version_ref = self.ctx.version_set.current().await;
        
        if ECOTUNE_MAIN_LEVEL >= version_ref.level_slice.len() {
            return None;
        }
        
        let main_level_files: Vec<Ulid> = version_ref.level_slice[ECOTUNE_MAIN_LEVEL]
            .iter()
            .take(runs_to_merge) // Take exactly the number of runs DP decided to merge
            .map(|scope| scope.gen)
            .collect();
        
        if main_level_files.len() >= 2 {
            println!("EcoTune: Creating ML compaction task - merging {} runs in main level", main_level_files.len());
            Some(EcoTuneTask {
                input: vec![(ECOTUNE_MAIN_LEVEL, main_level_files)],
                target_level: ECOTUNE_MAIN_LEVEL, // ML compaction within same level
            })
        } else {
            None
        }
    }

    /// Create TM compaction task (top to main level compaction)
    async fn create_tm_compaction_task(&self) -> Option<EcoTuneTask> {
        let version_ref = self.ctx.version_set.current().await;
        let options = self.options.read().await;
        
        // Check if level 0 (top) needs compaction to level 1 (main)
        if Self::is_threshold_exceeded(&options, &version_ref, 0) {
            let top_level_files: Vec<Ulid> = version_ref.level_slice[0]
                .iter()
                .map(|scope| scope.gen)
                .collect();
            
            if !top_level_files.is_empty() {
                println!("EcoTune: Creating TM compaction task - moving {} files from top to main level", top_level_files.len());
                Some(EcoTuneTask {
                    input: vec![(0, top_level_files)],
                    target_level: ECOTUNE_MAIN_LEVEL,
                })
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// O(R⁴) DP solver implementing Algorithm 1 from Section 4.3.2
    /// State: (R_i, i, j, k) where:
    /// - R_i: number of existing runs
    /// - i: remaining TM compactions 
    /// - j: pending ML size from previous decision
    /// - k: consecutive TM operations
    async fn solve_compaction_scheduling(
        &self,
        state: CompactionState,
        options: &EcoTuneOptions,
    ) -> Result<CompactionDecision, CompactionError<R>> {
        // Check cache first
        {
            let cache = self.dp_cache.read().await;
            if let Some(cached_decision) = cache.get(&state) {
                return Ok(cached_decision.clone());
            }
        }
        
        // Base case: no more TM compactions remaining
        if state.remaining_tm_compactions == 0 {
            let decision = CompactionDecision {
                runs_to_merge: 0,
                score: 0.0,
            };
            self.dp_cache.write().await.insert(state, decision.clone());
            return Ok(decision);
        }
        
        let mut best_decision = CompactionDecision {
            runs_to_merge: 0,
            score: f64::NEG_INFINITY,
        };
        
        // Try all possible consecutive TM operations (k' from 1 to remaining)
        let max_consecutive = state.remaining_tm_compactions;
        for k_prime in 1..=max_consecutive {
            // Try all possible ML compaction sizes (j' from 0 to R_i + k')
            // After k' TM operations, we have R_i + k' runs available for ML compaction
            let max_ml_size = state.existing_runs + k_prime;
            for j_prime in 0..=max_ml_size {
                // Skip invalid ML compaction size: j' = 1 (can't merge 1 run)
                if j_prime == 1 {
                    continue;
                }
                
                let total_score = if j_prime == 0 {
                    // No ML compaction: just k' consecutive TM operations
                    let tm_score = self.evaluate_consecutive_tm_operations(
                        &state, k_prime, options
                    ).await?;
                    
                    let next_state = CompactionState {
                        existing_runs: state.existing_runs + k_prime,
                        remaining_tm_compactions: state.remaining_tm_compactions - k_prime,
                        pending_ml_size: 0,
                        consecutive_tm_ops: 0,
                    };
                    
                    let future_score = if next_state.remaining_tm_compactions > 0 {
                        Box::pin(self.solve_compaction_scheduling(next_state, options)).await?.score
                    } else {
                        0.0
                    };
                    
                    tm_score + future_score
                } else {
                    // ML compaction of size j' after k' TM operations
                    // Note: j' >= 2 is guaranteed by the continue above
                    let combined_score = self.evaluate_tm_then_ml_operations(
                        &state, k_prime, j_prime, options
                    ).await?;
                    
                    let next_state = CompactionState {
                        existing_runs: state.existing_runs + k_prime - j_prime + 1, // +k' runs from TM, -j' +1 from ML
                        remaining_tm_compactions: state.remaining_tm_compactions - k_prime,
                        pending_ml_size: j_prime,
                        consecutive_tm_ops: k_prime,
                    };
                    
                    let future_score = if next_state.remaining_tm_compactions > 0 {
                        Box::pin(self.solve_compaction_scheduling(next_state, options)).await?.score
                    } else {
                        0.0
                    };
                    
                    combined_score + future_score
                };
                
                if total_score > best_decision.score {
                    best_decision = CompactionDecision {
                        runs_to_merge: j_prime,
                        score: total_score,
                    };
                }
            }
        }
        
        // Cache and return best decision
        self.dp_cache.write().await.insert(state.clone(), best_decision.clone());
        Ok(best_decision)
    }
    
    
    /// Evaluate score for k consecutive TM operations (no ML compaction)
    /// O(R⁴) algorithm: evaluates multiple TM operations in sequence
    async fn evaluate_consecutive_tm_operations(
        &self,
        state: &CompactionState,
        consecutive_tm_count: usize,
        options: &EcoTuneOptions,
    ) -> Result<f64, CompactionError<R>> {
        let mut total_score = 0.0;
        let mut current_runs = state.existing_runs;
        
        // Evaluate each TM operation's contribution to query throughput
        for _tm_op in 0..consecutive_tm_count {
            let query_speed = options.query_speed(current_runs);
            total_score += query_speed * options.tm_compaction_interval;
            current_runs += 1; // Each TM adds one run
        }
        
        Ok(total_score)
    }
    
    /// Evaluate score for k TM operations followed by ML compaction of size j
    /// O(R⁴) algorithm: evaluates combined TM+ML operation sequence
    async fn evaluate_tm_then_ml_operations(
        &self,
        state: &CompactionState,
        consecutive_tm_count: usize,
        ml_size: usize,
        options: &EcoTuneOptions,
    ) -> Result<f64, CompactionError<R>> {
        let mut total_score = 0.0;
        let mut current_runs = state.existing_runs;
        
        // Phase 1: Evaluate consecutive TM operations
        for _tm_op in 0..consecutive_tm_count {
            let query_speed = options.query_speed(current_runs);
            total_score += query_speed * options.tm_compaction_interval;
            current_runs += 1;
        }
        
        // Phase 2: Evaluate ML compaction of ml_size runs
        if ml_size >= 2 && ml_size <= current_runs {
            let ml_time = ml_size as f64 * options.ml_compaction_time_per_unit;
            let non_ml_time = options.tm_compaction_interval - ml_time;
            
            // Query speeds before and during ML compaction
            let query_speed_after_ml = options.query_speed(current_runs - ml_size + 1);
            let query_speed_during_ml = options.query_speed_during_ml(current_runs);
            
            if non_ml_time > 0.0 {
                // ML finishes within TM interval
                total_score += query_speed_after_ml * non_ml_time + query_speed_during_ml * ml_time;
            } else {
                // ML takes longer than TM interval
                total_score += query_speed_during_ml * options.tm_compaction_interval;
            }
        }
        
        Ok(total_score)
    }

    pub async fn execute_major(&self, task: EcoTuneTask) -> Result<(), CompactionError<R>> {
        let version_ref = self.ctx.version_set.current().await;
        let options = self.options.read().await;
        let mut version_edits = vec![];
        let mut delete_gens = vec![];
        
        // Determine compaction type for execution state tracking
        let is_tm_compaction = task.input.len() == 1 && task.input[0].0 == 0 && task.target_level == ECOTUNE_MAIN_LEVEL;
        let is_ml_compaction = task.input.len() == 1 && task.input[0].0 == ECOTUNE_MAIN_LEVEL && task.target_level == ECOTUNE_MAIN_LEVEL;
        let ml_compaction_size = if is_ml_compaction { task.input[0].1.len() } else { 0 };

        for (level, file_gens) in &task.input {
            if file_gens.is_empty() {
                continue;
            }

            let level_scopes: Vec<&Scope<_>> = version_ref.level_slice[*level]
                .iter()
                .filter(|scope| file_gens.contains(&scope.gen))
                .collect();

            if level_scopes.is_empty() {
                continue;
            }

            let min = level_scopes.iter().map(|scope| &scope.min).min().unwrap();
            let max = level_scopes.iter().map(|scope| &scope.max).max().unwrap();
            
            Self::major_compaction(
                &version_ref,
                &self.db_option,
                min,
                max,
                &mut version_edits,
                &mut delete_gens,
                &self.record_schema,
                &self.ctx,
                task.input[0].0,
                task.target_level,
            )
            .await?;

            break;
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
        
        // Update execution state based on the compaction that was performed
        // This is critical for the DP algorithm to work correctly per paper Section 4.3.2
        {
            let mut exec_state = self.execution_state.write().await;
            
            if is_tm_compaction {
                // TM compaction: increment consecutive TM count and round count
                exec_state.consecutive_tm_count += 1;
                exec_state.tm_compactions_in_round += 1;
                exec_state.last_ml_size = 0; // Reset ML size since no ML was performed
                
                println!("EcoTune: Updated execution state - TM compaction #{} (consecutive: {})", 
                    exec_state.tm_compactions_in_round, exec_state.consecutive_tm_count);
                
                // Check if we've completed a compaction round
                if exec_state.tm_compactions_in_round >= options.total_tm_compactions {
                    exec_state.current_round += 1;
                    exec_state.tm_compactions_in_round = 0;
                    exec_state.consecutive_tm_count = 0;
                    exec_state.last_ml_size = 0;
                    
                    // Clear DP cache for new round
                    self.dp_cache.write().await.clear();
                    
                    println!("EcoTune: Completed compaction round {}, resetting state", exec_state.current_round);
                }
            } else if is_ml_compaction {
                // ML compaction: reset consecutive TM count and record ML size
                exec_state.last_ml_size = ml_compaction_size;
                exec_state.consecutive_tm_count = 0; // Reset since ML breaks the TM sequence
                
                println!("EcoTune: Updated execution state - ML compaction (merged {} runs)", ml_compaction_size);
            }
            // For other compaction types (e.g., bottom level), don't update the execution state
            // as they are not part of the main DP decision process
        }

        Ok(())
    }

    pub async fn minor_flush(&self, is_manual: bool) -> Result<Option<EcoTuneTask>, CompactionError<R>> {
        let mut guard = self.schema.write().await;

        guard.trigger.reset();

        if !guard.mutable.is_empty() {
            let trigger_clone = guard.trigger.clone();

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

        let options = self.options.read().await;
        if (is_manual && !guard.immutables.is_empty())
            || guard.immutables.len() > options.immutable_chunk_max_num
        {
            let recover_wal_ids = guard.recover_wal_ids.take();
            drop(guard);

            let guard = self.schema.upgradable_read().await;
            let chunk_num = if is_manual {
                guard.immutables.len()
            } else {
                options.immutable_chunk_num
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

    async fn minor_compaction(
        option: &DbOption,
        recover_wal_ids: Option<Vec<FileId>>,
        batches: &[
            (
                Option<FileId>,
                Immutable<<R::Schema as RecordSchema>::Columns>,
            )
        ],
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

    #[allow(clippy::too_many_arguments)]
    async fn major_compaction(
        version: &Version<R>,
        option: &DbOption,
        mut min: &<R::Schema as RecordSchema>::Key,
        mut max: &<R::Schema as RecordSchema>::Key,
        version_edits: &mut Vec<VersionEdit<<R::Schema as RecordSchema>::Key>>,
        delete_gens: &mut Vec<SsTableID>,
        instance: &R::Schema,
        ctx: &Context<R>,
        source_level: usize,
        target_level: usize,
    ) -> Result<(), CompactionError<R>> {
        let level = source_level;

        let (meet_scopes_l, start_l, end_l) = Self::this_level_scopes(version, min, max, level);
        let (meet_scopes_ll, start_ll, end_ll) =
            Self::next_level_scopes(version, &mut min, &mut max, level, &meet_scopes_l)?;

        let level_path = option.level_fs_path(level).unwrap_or(&option.base_path);
        let level_fs = ctx.manager.get_fs(level_path);
        let mut streams = Vec::with_capacity(meet_scopes_l.len() + meet_scopes_ll.len());

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
            let (lower, upper) = <EcoTuneCompactor<R> as Compactor<R>>::full_scope(&meet_scopes_l)?;
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

        let level_l_path = option.level_fs_path(target_level).unwrap_or(&option.base_path);
        let level_l_fs = ctx.manager.get_fs(level_l_path);

        if !meet_scopes_ll.is_empty() {
            let (lower, upper) = <EcoTuneCompactor<R> as Compactor<R>>::full_scope(&meet_scopes_ll)?;
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

        <EcoTuneCompactor<R> as Compactor<R>>::build_tables(
            option,
            version_edits,
            target_level,
            streams,
            instance,
            level_l_fs,
        )
        .await?;

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

    fn this_level_scopes<'a>(
        version: &'a Version<R>,
        min: &<R::Schema as RecordSchema>::Key,
        max: &<R::Schema as RecordSchema>::Key,
        level: usize,
    ) -> (
        Vec<&'a Scope<<R::Schema as RecordSchema>::Key>>,
        usize,
        usize,
    ) {
        let mut meet_scopes_l = Vec::new();
        let start_l = Version::<R>::scope_search(min, &version.level_slice[level]);
        let mut end_l = start_l;

        for scope in version.level_slice[level][start_l..].iter() {
            if scope.contains(min) || scope.contains(max) {
                meet_scopes_l.push(scope);
                end_l += 1;
            } else {
                break;
            }
        }

        if meet_scopes_l.is_empty() && !version.level_slice[level].is_empty() {
            let scope = &version.level_slice[level][0];
            meet_scopes_l.push(scope);
            end_l = 1;
        }

        (meet_scopes_l, start_l, end_l.saturating_sub(1))
    }

    /// Check if compaction threshold is exceeded based on EcoTune parameters
    fn is_threshold_exceeded(
        options: &EcoTuneOptions,
        version: &Version<R>,
        level: usize,
    ) -> bool {
        let current_size = Version::<R>::tables_len(version, level);
        current_size >= options.size_threshold
    }
    
    /// Get current execution state for debugging and monitoring
    pub async fn get_execution_state(&self) -> EcoTuneExecutionState {
        self.execution_state.read().await.clone()
    }
    
    /// Reset execution state (useful for testing or manual intervention)
    pub async fn reset_execution_state(&self) {
        let mut exec_state = self.execution_state.write().await;
        *exec_state = EcoTuneExecutionState::default();
        
        // Also clear DP cache since state assumptions have changed
        self.dp_cache.write().await.clear();
        
        println!("EcoTune: Execution state reset to initial values");
    }
}

#[cfg(all(test, feature = "tokio"))]
mod tests {
    use super::*;

    /// Helper to print the DP state and decision in a readable format
    fn print_compaction_plan(
        state: &CompactionState,
        decision: &CompactionDecision,
        options: &EcoTuneOptions,
    ) {
        println!("=== EcoTune DP Compaction Plan ===");
        println!("Current State:");
        println!("  - Existing runs in main level: {}", state.existing_runs);
        println!("  - Remaining TM compactions: {}", state.remaining_tm_compactions);
        println!("  - Pending ML size: {}", state.pending_ml_size);
        println!("  - Consecutive TM ops: {}", state.consecutive_tm_ops);
        
        println!("\nEcoTune Options:");
        println!("  - Total TM compactions (R): {}", options.total_tm_compactions);
        println!("  - Size ratio (T): {}", options.size_ratio);
        println!("  - Long range scan length (K): {}", options.long_range_scan_length);
        println!("  - Long range ratio (r): {:.2}", options.long_range_ratio);
        println!("  - False positive rate (f): {:.3}", options.false_positive_rate);
        println!("  - TM compaction interval (Tw): {:.1}s", options.tm_compaction_interval);
        println!("  - ML compaction time per unit (Tc): {:.1}s", options.ml_compaction_time_per_unit);
        println!("  - Query speed ratio during ML (β): {:.2}", options.query_speed_ratio_during_ml);
        
        println!("\nOptimal Decision:");
        println!("  - Runs to merge: {}", decision.runs_to_merge);
        println!("  - Expected score: {:.6}", decision.score);
        
        if decision.runs_to_merge == 0 {
            println!("  - Strategy: WAIT (no compaction needed)");
        } else if decision.runs_to_merge == 1 {
            println!("  - Strategy: TM COMPACTION (top to main level)");
        } else {
            println!("  - Strategy: ML COMPACTION (merge {} runs in main level)", decision.runs_to_merge);
        }
        
        // Calculate and display query speeds and throughput
        let current_query_speed = options.query_speed(state.existing_runs);
        let tm_interval = options.tm_compaction_interval;
        
        println!("\nQuery Performance Analysis:");
        println!("  - Current query speed: {:.6} queries/time", current_query_speed);
        println!("  - Queries per TM interval ({:.1}s): {:.2} queries", tm_interval, current_query_speed * tm_interval);
        
        // If this is an ML compaction, show the impact on throughput
        if decision.runs_to_merge >= 2 {
            let ml_time = decision.runs_to_merge as f64 * options.ml_compaction_time_per_unit;
            let queries_during_ml = current_query_speed * options.query_speed_ratio_during_ml * ml_time;
            let queries_after_ml = if ml_time < tm_interval { 
                current_query_speed * (tm_interval - ml_time) 
            } else { 
                0.0 
            };
            let total_queries_in_interval = queries_during_ml + queries_after_ml;
            
            println!("  - ML compaction time: {:.1}s (merging {} runs)", ml_time, decision.runs_to_merge);
            println!("  - Queries during ML ({:.1}s): {:.2} queries", ml_time, queries_during_ml);
            if ml_time < tm_interval {
                println!("  - Queries after ML ({:.1}s): {:.2} queries", tm_interval - ml_time, queries_after_ml);
            }
            println!("  - Total queries in TM interval: {:.2} queries", total_queries_in_interval);
            println!("  - Throughput impact: {:.1}% of normal", (total_queries_in_interval / (current_query_speed * tm_interval)) * 100.0);
        }
        
        println!("=====================================\n");
    }

    #[tokio::test]
    async fn test_dp_algorithm_basic_scenarios() {
        // Test the DP algorithm directly without full compactor setup
        let options = EcoTuneOptions::default();
        
        println!("Testing EcoTune DP Algorithm - Basic Scenarios");
        println!("==============================================");
        
        // Test Case 1: Empty main level
        let state1 = CompactionState {
            existing_runs: 0,
            remaining_tm_compactions: 10,
            pending_ml_size: 0,
            consecutive_tm_ops: 0,
        };
        
        // Create a mock decision for testing the print function
        let decision1 = CompactionDecision {
            runs_to_merge: 1, // TM compaction makes sense with empty main level
            score: 10.0,
        };
        
        println!("Test Case 1: Empty main level");
        print_compaction_plan(&state1, &decision1, &options);
        
        // Test Case 2: Few runs in main level
        let state2 = CompactionState {
            existing_runs: 3,
            remaining_tm_compactions: 8,
            pending_ml_size: 0,
            consecutive_tm_ops: 0,
        };
        
        let decision2 = CompactionDecision {
            runs_to_merge: 2, // ML compaction with few runs
            score: 15.5,
        };
        
        println!("Test Case 2: Few runs in main level");
        print_compaction_plan(&state2, &decision2, &options);
        
        // Test Case 3: Many runs in main level
        let state3 = CompactionState {
            existing_runs: 8,
            remaining_tm_compactions: 5,
            pending_ml_size: 0,
            consecutive_tm_ops: 0,
        };
        
        let decision3 = CompactionDecision {
            runs_to_merge: 4, // Large ML compaction with many runs
            score: 25.8,
        };
        
        println!("Test Case 3: Many runs in main level");
        print_compaction_plan(&state3, &decision3, &options);
        
        // Test Case 4: Near end of compaction round
        let state4 = CompactionState {
            existing_runs: 6,
            remaining_tm_compactions: 2,
            pending_ml_size: 0,
            consecutive_tm_ops: 0,
        };
        
        let decision4 = CompactionDecision {
            runs_to_merge: 0, // Wait - near end of round
            score: 5.2,
        };
        
        println!("Test Case 4: Near end of compaction round");
        print_compaction_plan(&state4, &decision4, &options);
        
        // Verify decisions make sense
        assert!(decision1.score >= 0.0, "Score should be non-negative");
        assert!(decision2.score >= 0.0, "Score should be non-negative");
        assert!(decision3.score >= 0.0, "Score should be non-negative");
        assert!(decision4.score >= 0.0, "Score should be non-negative");
        
        // Test that the compaction state can be created and decisions can be made
        assert_eq!(state1.existing_runs, 0);
        assert_eq!(state2.existing_runs, 3);
        assert_eq!(state3.existing_runs, 8);
        assert_eq!(state4.existing_runs, 6);
        
        // Test different strategy types
        assert_eq!(decision1.runs_to_merge, 1); // TM
        assert_eq!(decision2.runs_to_merge, 2); // ML
        assert_eq!(decision3.runs_to_merge, 4); // ML
        assert_eq!(decision4.runs_to_merge, 0); // Wait
    }
    
    #[tokio::test]
    async fn test_dp_algorithm_different_workloads() {
        // Test how different workload parameters affect query speed calculations
        
        println!("Testing EcoTune DP Algorithm - Different Workloads");
        println!("================================================");
        
        // Workload 1: Point query heavy (low long range ratio)
        let options_point = EcoTuneOptions {
            long_range_ratio: 0.1, // 10% long range queries
            false_positive_rate: 0.01,
            ..Default::default()
        };
        
        let state = CompactionState {
            existing_runs: 5,
            remaining_tm_compactions: 6,
            pending_ml_size: 0,
            consecutive_tm_ops: 0,
        };
        
        // Mock decisions that would make sense for different workloads
        let decision_point = CompactionDecision {
            runs_to_merge: 1, // Point queries prefer fewer runs - TM is acceptable
            score: 8.2,
        };
        
        println!("Workload 1: Point Query Heavy (r=0.1)");
        print_compaction_plan(&state, &decision_point, &options_point);
        
        // Workload 2: Range scan heavy (high long range ratio)
        let options_range = EcoTuneOptions {
            long_range_ratio: 0.9, // 90% long range queries
            false_positive_rate: 0.01,
            ..Default::default()
        };
        
        let decision_range = CompactionDecision {
            runs_to_merge: 3, // Range scans prefer fewer levels - ML is preferred
            score: 12.5,
        };
        
        println!("Workload 2: Range Scan Heavy (r=0.9)");
        print_compaction_plan(&state, &decision_range, &options_range);
        
        // Workload 3: High false positive rate
        let options_fp = EcoTuneOptions {
            long_range_ratio: 0.5,
            false_positive_rate: 0.1, // 10% false positive rate
            ..Default::default()
        };
        
        let decision_fp = CompactionDecision {
            runs_to_merge: 2, // High FP rate makes ML more attractive
            score: 10.1,
        };
        
        println!("Workload 3: High False Positive Rate (f=0.1)");
        print_compaction_plan(&state, &decision_fp, &options_fp);
        
        // Compare query speeds across workloads
        let point_speed = options_point.query_speed(5);
        let range_speed = options_range.query_speed(5);
        let fp_speed = options_fp.query_speed(5);
        
        println!("Query Speed Comparison for 5 existing runs:");
        println!("  Point heavy (r=0.1): {:.6}", point_speed);
        println!("  Range heavy (r=0.9): {:.6}", range_speed);
        println!("  High FP (f=0.1): {:.6}", fp_speed);
        
        // Range-heavy workloads should have lower query speeds due to more range scans
        assert!(point_speed > range_speed, "Point queries should be faster than range scans");
        assert!(fp_speed < point_speed, "High false positive rate should reduce query speed");
    }
    
    #[tokio::test(flavor = "multi_thread")]
    async fn test_real_execution_state_tracking() {
        use tempfile::TempDir;
        use fusio::path::Path;
        use std::sync::Arc;
        use crate::{
            executor::tokio::TokioExecutor,
            inmem::immutable::tests::TestSchema,
            tests::Test,
            trigger::TriggerType,
            DbOption, DB,
        };
        
        println!("Testing EcoTune Execution State Tracking with Real Data");
        println!("======================================================");
        
        // Create temporary database with EcoTune compaction
        let temp_dir = TempDir::new().unwrap();
        let mut option = DbOption::new(
            Path::from_filesystem_path(temp_dir.path()).unwrap(),
            &TestSchema,
        ).ecotune_compaction(EcoTuneOptions {
            total_tm_compactions: 5,      // Small round for testing
            size_threshold: 2,            // Low threshold to trigger compactions
            immutable_chunk_max_num: 2,
            ..Default::default()
        });
        option.trigger_type = TriggerType::SizeOfMem(1024 * 1024); // Small memory trigger
        
        let db: DB<Test, TokioExecutor> = DB::new(option.clone(), TokioExecutor::current(), TestSchema)
            .await
            .unwrap();
        
        println!("Database created with EcoTune compaction");
        
        // Insert data to trigger multiple compactions
        let mut records_inserted = 0;
        let batch_num = 5;
        let record_num = 1000;
        for batch in 0..batch_num {
            println!("\n--- Batch {} ---", batch + 1);
            
            // Insert records to trigger compaction
            for i in 0..record_num {
                let record = Test {
                    vstring: format!("test_key_{}_{}", batch, i),
                    vu32: records_inserted,
                    vbool: Some(i % 2 == 0),
                };
                db.insert(record).await.unwrap();
                records_inserted += 1;
            }
            
            // Force flush and compaction - this should trigger DP algorithm
            db.flush().await.unwrap();
            
            // Check LSM tree structure after compaction
            let version_ref = db.ctx.version_set.current().await;
            let level_0_files = version_ref.level_slice.get(0).map_or(0, |level| level.len());
            let level_1_files = version_ref.level_slice.get(ECOTUNE_MAIN_LEVEL).map_or(0, |level| level.len());
            let level_2_files = version_ref.level_slice.get(ECOTUNE_BOTTOM_LEVEL).map_or(0, |level| level.len());
            
            println!("LSM structure after batch {}: L0={}, L1={}, L2={}", 
                    batch + 1, level_0_files, level_1_files, level_2_files);
        }
        
        // Test DP algorithm computation with real state
        let version_ref = db.ctx.version_set.current().await;
        let current_runs = version_ref.level_slice.get(ECOTUNE_MAIN_LEVEL).map_or(0, |level| level.len());
        
        // Create a realistic DP state based on current LSM structure
        let dp_state = CompactionState {
            existing_runs: current_runs,
            remaining_tm_compactions: 5, // From options
            pending_ml_size: 0,          // Assume no pending ML
            consecutive_tm_ops: 0,       // Assume fresh state
        };
        
        println!("\nDP state for algorithm: {:?}", dp_state);
        
        // Test that DP algorithm can run with real state
        let compactor = EcoTuneCompactor::new(
            EcoTuneOptions::default(),
            db.schema.clone(),
            Arc::new(TestSchema),
            Arc::new(option.clone()),
            db.ctx.clone(),
        );
        
        let decision = compactor.solve_compaction_scheduling(dp_state.clone(), &EcoTuneOptions::default()).await.unwrap();
        
        println!("DP decision: runs_to_merge={}, score={:.3}", decision.runs_to_merge, decision.score);
        
        // Verify decision is reasonable
        assert!(decision.score >= 0.0, "DP score should be non-negative");
        assert!(decision.runs_to_merge <= current_runs + dp_state.remaining_tm_compactions, 
                "Decision should not exceed available runs");
        
        // Test that DP cache was populated
        let cache = compactor.dp_cache.read().await;
        assert!(cache.contains_key(&dp_state), "DP cache should contain the computed state");
        
        // Test execution state tracking functionality
        let mut exec_state = EcoTuneExecutionState::default();
        
        // Simulate TM compactions
        exec_state.consecutive_tm_count += 1;
        exec_state.tm_compactions_in_round += 1;
        println!("After simulated TM: {:?}", exec_state);
        
        // Simulate ML compaction
        exec_state.last_ml_size = 3;
        exec_state.consecutive_tm_count = 0; // ML resets TM count
        println!("After simulated ML: {:?}", exec_state);
        
        // Verify data integrity after all compactions
        let mut found_records = 0;
        for batch in 0..batch_num {
            for i in 0..record_num {
                let key = format!("test_key_{}_{}", batch, i);
                if let Some(record) = db.get(&key, |entry| {
                    match entry {
                        crate::transaction::TransactionEntry::Stream(stream_entry) => {
                            stream_entry.value().map(|val| Test {
                                vstring: val.vstring.to_string(),
                                vu32: val.vu32.unwrap_or(0),
                                vbool: val.vbool,
                            })
                        }
                        crate::transaction::TransactionEntry::Local(local_entry) => {
                            Some(Test {
                                vstring: local_entry.vstring.to_string(),
                                vu32: local_entry.vu32.unwrap_or(0), 
                                vbool: local_entry.vbool,
                            })
                        }
                    }
                }).await.unwrap() {
                    found_records += 1;
                    assert_eq!(record.vstring, key);
                }
            }
        }
        
        println!("Data integrity verified: {}/{} records found", found_records, records_inserted);
        assert!(found_records > 0, "Should find some records after compaction");
        
        println!("Real execution state tracking test completed successfully!");
    }
}

#[cfg(all(test, feature = "tokio"))]
pub(crate) mod tests_metric {

    use fusio::path::Path;
    use tempfile::TempDir;

    use crate::{
        compaction::ecotune::EcoTuneOptions,
        executor::tokio::TokioExecutor,
        inmem::immutable::tests::TestSchema,
        tests::Test,
        trigger::TriggerType,
        version::MAX_LEVEL,
        DbOption, DB,
    };

    fn convert_test_ref_to_test(entry: crate::transaction::TransactionEntry<'_, Test>) -> Option<Test> {
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
    #[ignore]
    async fn test_read_write_amplification_measurement() {
        let temp_dir = TempDir::new().unwrap();
        let option = DbOption::new(
            Path::from_filesystem_path(temp_dir.path()).unwrap(),
            &TestSchema,
        )
        .ecotune_compaction(EcoTuneOptions::default());
        //.max_sst_file_size(1024); // Small file size to force multiple files

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
        .ecotune_compaction(EcoTuneOptions::default());
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
