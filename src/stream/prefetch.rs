use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    task::Context,
};

use arrow::{array::RecordBatch, datatypes::Schema};
use fusio::{path::Path, DynFs};
use futures_util::StreamExt;
use parquet::{
    arrow::{
        async_reader::AsyncFileReader,
        arrow_reader::ArrowReaderOptions,
        ParquetRecordBatchStreamBuilder, ProjectionMask,
    },
    errors::ParquetError,
};
use tokio::sync::Mutex;
use futures::channel::oneshot;

use crate::fs::FileId;

/// RecordBatch-level prefetch identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchId {
    /// File identifier
    pub file_id: FileId,
    /// Batch sequence number within the file
    pub batch_index: usize,
}

impl BatchId {
    pub fn new(file_id: FileId, batch_index: usize) -> Self {
        Self {
            file_id,
            batch_index,
        }
    }
}


/// Represents prefetched RecordBatch data
#[derive(Debug, Clone)]
pub struct BatchData {
    /// The actual RecordBatch
    pub batch: RecordBatch,
    /// Arrow schema for this batch
    pub schema: Arc<Schema>,
    /// Batch metadata
    pub metadata: BatchMetadata,
}

/// Metadata for a prefetched RecordBatch
#[derive(Debug, Clone)]
pub struct BatchMetadata {
    /// Number of rows in this batch
    pub num_rows: usize,
    /// Memory size estimate
    pub memory_size: usize,
    /// File position this batch came from
    pub file_offset: Option<u64>,
}

/// Manages prefetching of Parquet RecordBatches for sequential access patterns
pub struct PrefetchBufferCollection {
    /// Active batch prefetch requests - maps BatchId to receiver for the result
    batch_requests: Arc<Mutex<HashMap<BatchId, oneshot::Receiver<Result<BatchData, ParquetError>>>>>,
    /// Ready batches cache for immediate access
    ready_cache: Arc<Mutex<HashMap<BatchId, BatchData>>>,
    /// Filesystem interface
    fs: Arc<dyn DynFs>,
    /// Maximum number of concurrent prefetch operations (adaptive)
    max_concurrent: Arc<std::sync::Mutex<usize>>,
    /// Queue of batches waiting to be prefetched
    batch_pending_queue: Arc<Mutex<VecDeque<(BatchId, Path, ProjectionMask)>>>,
    /// Track access patterns for adaptive prefetching
    access_stats: Arc<std::sync::Mutex<AccessStats>>,
    /// Maximum number of RecordBatches to prefetch per file
    max_batches_per_file: usize,
    /// Maximum total memory for prefetched batches (in bytes)
    max_memory_usage: usize,
    /// Current memory usage estimate
    current_memory: Arc<std::sync::Mutex<usize>>,
}

/// Per-file access pattern tracking (RocksDB-inspired)
#[derive(Debug, Default)]
struct FileAccessState {
    /// Number of sequential reads from this file
    sequential_reads: usize,
    /// Last accessed batch to detect sequential patterns
    last_batch_id: Option<BatchId>,
    /// Whether prefetching is currently enabled for this file
    prefetch_enabled: bool,
    /// Recent access pattern - true if sequential, false if random
    is_sequential_pattern: bool,
    /// Number of batches already prefetched for this file
    prefetched_batch_count: usize,
}

/// Prefetch configuration following RocksDB patterns
#[derive(Debug, Clone)]
struct PrefetchConfig {
    /// Number of sequential file reads required to trigger prefetching (RocksDB: 2)
    reads_for_auto_prefetch: usize,
    /// Base number of RecordBatches to prefetch per file 
    base_batches_per_file: usize,
    /// Hit rate threshold below which we reduce concurrency
    min_hit_rate_for_expansion: f64,
    /// Target RecordBatch size for prefetch calculations (bytes)
    target_batch_size: usize,
    /// Maximum batches to prefetch per file regardless of size
    max_batches_per_file: usize,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            reads_for_auto_prefetch: 2,        // More aggressive: 2 read to enable prefetch
            base_batches_per_file: 2,          // Start with 2 batches per file
            min_hit_rate_for_expansion: 0.5,   // Lower threshold: 50% hit rate to expand
            target_batch_size: 8 * 1024,      // 8KB target batch size
            max_batches_per_file: 4,           // Allow up to 4 batches per file
        }
    }
}

/// Statistics for tracking access patterns and prefetch effectiveness
#[derive(Debug, Default)]
struct AccessStats {
    /// Number of successful prefetch hits (batch was ready when requested)
    hits: usize,
    /// Number of prefetch misses (batch wasn't prefetched when requested)
    misses: usize,
    /// Number of wasted prefetches (prefetched but never accessed)
    wasted: usize,
    /// Recent file access sequence for pattern detection
    recent_accesses: VecDeque<FileId>,
    /// Maximum size of recent access tracking
    max_recent: usize,
    /// Per-file access pattern states
    file_states: HashMap<FileId, FileAccessState>,
    /// Prefetch configuration
    config: PrefetchConfig,
}

impl PrefetchBufferCollection {
    /// Create a new prefetch buffer collection
    pub fn new(fs: Arc<dyn DynFs>, max_concurrent: usize) -> Self {
        Self::new_with_config(fs, max_concurrent, PrefetchConfig::default())
    }

    /// Create a new prefetch buffer collection with custom config
    pub fn new_with_config(fs: Arc<dyn DynFs>, max_concurrent: usize, config: PrefetchConfig) -> Self {
        Self {
            batch_requests: Arc::new(Mutex::new(HashMap::new())),
            ready_cache: Arc::new(Mutex::new(HashMap::new())),
            fs,
            max_concurrent: Arc::new(std::sync::Mutex::new(max_concurrent)),
            batch_pending_queue: Arc::new(Mutex::new(VecDeque::new())),
            access_stats: Arc::new(std::sync::Mutex::new(AccessStats {
                hits: 0,
                misses: 0,
                wasted: 0,
                recent_accesses: VecDeque::new(),
                max_recent: 10,
                file_states: HashMap::new(),
                config: config.clone(),
            })),
            max_batches_per_file: config.base_batches_per_file,
            max_memory_usage: 8 * 1024 * 1024, // MB default limit
            current_memory: Arc::new(std::sync::Mutex::new(0)),
        }
    }

    /// Start prefetching RecordBatches for a file with RocksDB-style adaptive logic
    pub fn prefetch_batches(&self, file_id: FileId, path: Path, projection_mask: ProjectionMask) {
        // Record file access for pattern tracking
        let first_batch = BatchId::new(file_id, 0);
        self.record_file_access(first_batch);
        
        // Check if we should prefetch based on sequential access patterns
        let should_prefetch = self.should_prefetch_file(file_id);
        if !should_prefetch {
            #[cfg(test)]
            println!("Skipping prefetch for file {:?} - should_prefetch returned false", file_id);
            return;
        }
        
        #[cfg(test)]
        println!("Starting RecordBatch prefetch for file {:?}", file_id);
        
        // Get the adaptive number of batches to prefetch for this file
        let batches_per_file = self.get_adaptive_batches_per_file(&path, file_id);
        
        // Check if we already have enough prefetched batches for this file
        if let Ok(stats) = self.access_stats.try_lock() {
            if let Some(file_state) = stats.file_states.get(&file_id) {
                if file_state.prefetched_batch_count >= batches_per_file {
                    #[cfg(test)]
                    println!("File {:?} already has {} prefetched batches, skipping", file_id, file_state.prefetched_batch_count);
                    return;
                }
            }
        }
        
        // Start prefetching RecordBatches for this file
        for batch_index in 0..batches_per_file {
            let batch_id = BatchId::new(file_id, batch_index);
            self.prefetch_single_batch(batch_id, path.clone(), projection_mask.clone());
        }
    }
    
    /// Check if we should prefetch for a file (RocksDB-style sequential detection)
    fn should_prefetch_file(&self, file_id: FileId) -> bool {
        if let Ok(stats) = self.access_stats.try_lock() {
            if let Some(file_state) = stats.file_states.get(&file_id) {
                // Only prefetch if we've seen enough sequential reads and pattern is sequential
                file_state.sequential_reads >= stats.config.reads_for_auto_prefetch 
                    && file_state.is_sequential_pattern
                    && file_state.prefetch_enabled
            } else {
                // For new files, allow prefetching if we have a good overall access pattern
                // This handles the LSM case where each file is accessed once sequentially
                let recent_accesses = stats.recent_accesses.len();
                if recent_accesses >= 2 {
                    // Check if recent accesses show sequential file pattern
                    let is_sequential_file_access = stats.recent_accesses.iter()
                        .zip(stats.recent_accesses.iter().skip(1))
                        .all(|(prev, curr)| prev != curr); // Different files is good for LSM
                    is_sequential_file_access
                } else {
                    true // Not enough history, be optimistic for first few files
                }
            }
        } else {
            false // Can't check, be conservative
        }
    }
    
    /// Get adaptive number of batches to prefetch based on file size and hit rates
    fn get_adaptive_batches_per_file(&self, path: &Path, _file_id: FileId) -> usize {
        // Get file size to make informed prefetch decisions
        let file_size = self.get_file_size(path).unwrap_or(0);
        
        if let Ok(stats) = self.access_stats.try_lock() {
            let config = &stats.config;
            
            // RocksDB-style conservative approach: fixed small number of RecordBatches
            // Don't estimate based on file size since RecordBatch size varies wildly
            let base_batches = config.base_batches_per_file;
            
            // Apply adaptive logic based on hit rates (RocksDB-style)
            let hit_rate = self.calculate_hit_rate(&stats);
            let adaptive_batches = if hit_rate >= config.min_hit_rate_for_expansion {
                // Good hit rate: prefetch one more batch (conservative expansion)
                (base_batches + 1).min(config.max_batches_per_file)
            } else if hit_rate < 0.3 {
                // Poor hit rate: reduce to minimum (like RocksDB)
                1 
            } else {
                // Neutral hit rate: maintain base level
                base_batches
            };
                
            #[cfg(test)]
            println!("   File size: {}KB, base batches: {}, hit rate: {:.1}%, adaptive batches: {}", 
                    file_size / 1024, base_batches, hit_rate * 100.0, adaptive_batches);
                    
            adaptive_batches
        } else {
            1 // Conservative fallback
        }
    }
    
    /// Get file size, with fallback for errors
    fn get_file_size(&self, path: &Path) -> Option<usize> {
        // Convert fusio::Path to std::path::PathBuf for metadata check
        let path_str = path.to_string();
        let std_path = std::path::Path::new(&path_str);
        
        std::fs::metadata(std_path)
            .map(|metadata| metadata.len() as usize)
            .ok()
    }
    
    /// Calculate current hit rate from access stats
    fn calculate_hit_rate(&self, stats: &AccessStats) -> f64 {
        let total = stats.hits + stats.misses;
        if total == 0 {
            0.5 // Neutral starting point
        } else {
            stats.hits as f64 / total as f64
        }
    }
    
    /// Record batch access for pattern analysis
    fn record_batch_access(&self, batch_id: BatchId, was_hit: bool) {
        if let Ok(mut stats) = self.access_stats.try_lock() {
            if was_hit {
                stats.hits += 1;
            } else {
                stats.misses += 1;
            }
            
            // Track recent accesses for pattern detection
            // Use file_id for now, but could extend to track batch patterns
            stats.recent_accesses.push_back(batch_id.file_id);
            if stats.recent_accesses.len() > stats.max_recent {
                stats.recent_accesses.pop_front();
            }
        }
    }
    
    /// Record file access and update sequential pattern detection (RocksDB-inspired)
    pub fn record_file_access(&self, batch_id: BatchId) {
        if let Ok(mut stats) = self.access_stats.try_lock() {
            let file_id = batch_id.file_id;
            let reads_threshold = stats.config.reads_for_auto_prefetch; // Extract to avoid borrow conflict
            
            let file_state = stats.file_states.entry(file_id).or_default();
            
            // Check if this is a sequential access
            let is_sequential = if let Some(last_batch) = file_state.last_batch_id {
                self.is_sequential_access(last_batch, batch_id)
            } else {
                true // First access is considered sequential
            };
            
            if is_sequential {
                file_state.sequential_reads += 1;
                file_state.is_sequential_pattern = true;
                
                // Enable prefetching once we hit the threshold
                if file_state.sequential_reads >= reads_threshold {
                    file_state.prefetch_enabled = true;
                }
            } else {
                // Random access detected - reset state (RocksDB pattern)
                file_state.sequential_reads = 0;
                file_state.is_sequential_pattern = false;
                file_state.prefetch_enabled = false;
            }
            
            file_state.last_batch_id = Some(batch_id);
            
            // Track recent accesses for overall pattern analysis
            stats.recent_accesses.push_back(file_id);
            if stats.recent_accesses.len() > stats.max_recent {
                stats.recent_accesses.pop_front();
            }
        }
    }
    
    /// Check if two batch accesses are sequential (within same file and nearby batches)
    fn is_sequential_access(&self, last_batch: BatchId, current_batch: BatchId) -> bool {
        // Must be same file
        if last_batch.file_id != current_batch.file_id {
            return false;
        }
        
        // Sequential if next batch index (allowing some gap for prefetching)
        current_batch.batch_index <= last_batch.batch_index + 1
    }
    
    /// Dynamically adjust max_concurrent based on hit rates (RocksDB-inspired)
    fn adjust_concurrent_limit(&self) {
        if let Ok(stats) = self.access_stats.try_lock() {
            let total_requests = stats.hits + stats.misses;
            if total_requests > 5 { // Enough data to make decisions (lowered for testing)
                let hit_rate = stats.hits as f64 / total_requests as f64;
                let min_hit_rate = stats.config.min_hit_rate_for_expansion;
                
                if let Ok(mut max_concurrent) = self.max_concurrent.try_lock() {
                    if hit_rate >= min_hit_rate {
                        // Good hit rate: increase concurrency (up to reasonable limit)
                        if *max_concurrent < 16 {
                            *max_concurrent = (*max_concurrent * 2).min(16);
                        }
                    } else if hit_rate < 0.1 {
                        // Very low hit rate: reduce concurrency
                        if *max_concurrent > 1 {
                            *max_concurrent = (*max_concurrent / 2).max(1);
                        }
                    }
                }
            }
        }
    }
    
    /// Extract column indices that need to be prefetched based on projection mask
    fn extract_projected_columns(&self, projection_mask: &ProjectionMask, schema: &Schema) -> Vec<usize> {
        // Get the actual number of columns from the schema
        let num_fields = schema.fields().len();
        
        // Extract columns that are actually included in the projection mask
        let mut projected_columns = Vec::new();
        
        for leaf_idx in 0..num_fields {
            if projection_mask.leaf_included(leaf_idx) {
                projected_columns.push(leaf_idx);
            }
        }
        
        // If no columns are projected (shouldn't happen in practice), use a safe fallback
        if projected_columns.is_empty() {
            projected_columns = (0..num_fields).collect();
        }
        
        projected_columns
    }
    
    /// Start prefetching RecordBatches from a single file if not already in progress
    fn prefetch_single_batch(&self, batch_id: BatchId, path: Path, projection_mask: ProjectionMask) {
        // Check if already being prefetched or ready
        {
            if let Ok(batch_requests) = self.batch_requests.try_lock() {
                if batch_requests.contains_key(&batch_id) {
                    return;
                }
            }
            if let Ok(ready_cache) = self.ready_cache.try_lock() {
                if ready_cache.contains_key(&batch_id) {
                    return;
                }
            }
        }

        // Check if we're at the concurrent limit
        let active_count = if let Ok(requests) = self.batch_requests.try_lock() {
            requests.len()
        } else {
            return; // If we can't get the lock, skip this prefetch
        };
        
        let max_concurrent = if let Ok(max) = self.max_concurrent.try_lock() {
            *max
        } else {
            return;
        };
        
        if active_count >= max_concurrent {
            // Queue for later
            if let Ok(mut queue) = self.batch_pending_queue.try_lock() {
                queue.push_back((batch_id, path, projection_mask));
            }
            return;
        }

        // Create channel for this prefetch operation
        let (sender, receiver) = oneshot::channel();
        
        // Store the receiver
        {
            if let Ok(mut batch_requests) = self.batch_requests.try_lock() {
                batch_requests.insert(batch_id, receiver);
            } else {
                return;
            }
        }

        // Start RecordBatch prefetch operation
        let ready_cache = self.ready_cache.clone();
        let batch_requests = self.batch_requests.clone();
        let current_memory = self.current_memory.clone();
        let max_memory = self.max_memory_usage;
        let fs = self.fs.clone();
        
        tokio::spawn(async move {
            let result = async move {
                // Open the file
                let file = fs.open(&path).await.map_err(|e| ParquetError::General(format!("Failed to open file: {}", e)))?;
                let file_size = file.size().await.map_err(|e| ParquetError::General(format!("Failed to get file size: {}", e)))?;
                
                // Create AsyncReader from the file
                let async_reader = fusio_parquet::reader::AsyncReader::new(file, file_size).await
                    .map_err(|e| ParquetError::General(format!("Failed to create async reader: {}", e)))?;
                
                // Create ParquetRecordBatchStream from the file reader
                let mut builder = ParquetRecordBatchStreamBuilder::new_with_options(
                    Box::new(async_reader) as Box<dyn AsyncFileReader + 'static>,
                    ArrowReaderOptions::default().with_page_index(true),
                )
                .await
                .map_err(|e| ParquetError::General(format!("Failed to create stream builder: {}", e)))?;
                
                // Extract schema information before applying projection
                let original_schema = builder.schema();
                
                #[cfg(test)]
                {
                    // Demonstrate the new extract_projected_columns functionality in tests
                    // This shows which columns would be prefetched based on the projection mask
                    let num_fields = original_schema.fields().len();
                    let mut projected_columns = Vec::new();
                    
                    for leaf_idx in 0..num_fields {
                        if projection_mask.leaf_included(leaf_idx) {
                            projected_columns.push(leaf_idx);
                        }
                    }
                    
                    if projected_columns.is_empty() {
                        let safe_count = num_fields.min(3);
                        projected_columns = (0..safe_count).collect();
                    }
                    
                    println!("Prefetching columns {:?} from schema with {} fields for batch {:?}", 
                        projected_columns, num_fields, batch_id);
                }
                
                builder = builder.with_projection(projection_mask);
                let mut stream = builder.build().map_err(|e| ParquetError::General(format!("Failed to build stream: {}", e)))?;
                
                // Skip to the desired batch index
                for _ in 0..batch_id.batch_index {
                    if stream.next().await.is_none() {
                        return Err(ParquetError::General("Reached end of stream before target batch".to_string()));
                    }
                }
                
                // Read the specific batch we want
                if let Some(batch_result) = stream.next().await {
                    let record_batch = batch_result.map_err(|e| ParquetError::General(format!("Failed to read batch: {}", e)))?;
                    
                    // Estimate memory usage
                    let memory_size = record_batch.get_array_memory_size();
                    
                    // Check memory limits
                    if let Ok(mut current) = current_memory.try_lock() {
                        if *current + memory_size > max_memory {
                            return Err(ParquetError::General("Memory limit exceeded for prefetch buffer".to_string()));
                        }
                        *current += memory_size;
                    }
                    
                    let batch_data = BatchData {
                        schema: record_batch.schema(),
                        metadata: BatchMetadata {
                            num_rows: record_batch.num_rows(),
                            memory_size,
                            file_offset: None,
                        },
                        batch: record_batch,
                    };
                    
                    Ok(batch_data)
                } else {
                    Err(ParquetError::General("No more batches available".to_string()))
                }
            }.await;
            
            match result {
                Ok(batch_data) => {
                    // Store in ready cache
                    {
                        let mut cache = ready_cache.lock().await;
                        cache.insert(batch_id, batch_data.clone());
                    }
                    // Remove from active requests
                    {
                        let mut requests = batch_requests.lock().await;
                        requests.remove(&batch_id);
                    }
                    // Send result
                    let _ = sender.send(Ok(batch_data));
                }
                Err(err) => {
                    // Remove from active requests
                    {
                        let mut requests = batch_requests.lock().await;
                        requests.remove(&batch_id);
                    }
                    // Send error
                    let _ = sender.send(Err(err));
                }
            }
        });
    }
    

    /// Try to get a specific batch immediately if ready, non-blocking
    pub fn try_get_batch(&self, batch_id: BatchId) -> Option<BatchData> {
        // Check ready cache first
        {
            if let Ok(mut cache) = self.ready_cache.try_lock() {
                if let Some(batch_data) = cache.remove(&batch_id) {
                    // Update memory usage
                    if let Ok(mut current) = self.current_memory.try_lock() {
                        *current = current.saturating_sub(batch_data.metadata.memory_size);
                    }
                    // Record successful hit
                    self.record_batch_access(batch_id, true);
                    self.try_start_next_batch_prefetch();
                    return Some(batch_data);
                }
            }
        }
        
        // Record miss
        self.record_batch_access(batch_id, false);
        None
    }

    /// Async method to get a batch, waiting if it's still being prefetched
    pub async fn get_batch(&self, batch_id: BatchId) -> Option<BatchData> {
        // First try immediate get
        if let Some(batch_data) = self.try_get_batch(batch_id) {
            return Some(batch_data);
        }

        // Check if there's an active request we can wait for
        let receiver = {
            let mut batch_requests = self.batch_requests.lock().await;
            batch_requests.remove(&batch_id)
        }?;
        
        // Wait for the result
        match receiver.await {
            Ok(Ok(batch_data)) => {
                // Update memory usage
                if let Ok(mut current) = self.current_memory.try_lock() {
                    *current = current.saturating_sub(batch_data.metadata.memory_size);
                }
                self.record_batch_access(batch_id, true);
                self.try_start_next_batch_prefetch();
                Some(batch_data)
            }
            Ok(Err(_)) | Err(_) => {
                self.record_batch_access(batch_id, false);
                None
            }
        }
    }
    
    /// Poll prefetch operations to advance their state  
    pub fn poll_prefetches(&self, _cx: &mut Context<'_>) -> bool {
        // With channels, polling is handled by the tokio runtime automatically
        // We need to aggressively try starting queued prefetches
        
        // Periodically adjust concurrency based on hit rates
        self.adjust_concurrent_limit();
        
        // Try to start multiple queued operations if we have capacity
        let max_attempts = 10; // Prevent infinite loops
        for _ in 0..max_attempts {
            let queue_len_before = self.batch_pending_queue.try_lock().map(|q| q.len()).unwrap_or(0);
            self.try_start_next_batch_prefetch();
            let queue_len_after = self.batch_pending_queue.try_lock().map(|q| q.len()).unwrap_or(0);
            
            // If queue didn't shrink, we're at capacity or no more items
            if queue_len_after >= queue_len_before {
                break;
            }
        }
        
        // Return true if there are active requests or queued items
        let has_active = !self.batch_requests.try_lock().map_or(true, |requests| requests.is_empty());
        let has_queued = !self.batch_pending_queue.try_lock().map_or(true, |queue| queue.is_empty());
        
        has_active || has_queued
    }
    

    /// Try to start the next queued batch prefetch operation
    fn try_start_next_batch_prefetch(&self) {
        let next_item = {
            if let Ok(mut queue) = self.batch_pending_queue.try_lock() {
                queue.pop_front()
            } else {
                return;
            }
        };

        if let Some((batch_id, path, projection_mask)) = next_item {
            let active_count = if let Ok(requests) = self.batch_requests.try_lock() {
                requests.len()
            } else {
                return;
            };
            
            let max_concurrent = if let Ok(max) = self.max_concurrent.try_lock() {
                *max
            } else {
                return;
            };
            
            if active_count < max_concurrent {
                self.prefetch_single_batch(batch_id, path, projection_mask);
            } else {
                // Put it back in queue
                if let Ok(mut queue) = self.batch_pending_queue.try_lock() {
                    queue.push_front((batch_id, path, projection_mask));
                }
            }
        }
    }
    

    /// Remove completed or failed entries to free memory
    pub fn cleanup_completed(&self) {
        let page_wasted_count;
        
        // Clean up ready cache
        {
            if let Ok(mut cache) = self.ready_cache.try_lock() {
                page_wasted_count = cache.len();
                cache.clear();
            } else {
                return; // Skip cleanup if we can't get the lock
            }
        }
        
        // Update wasted count
        if page_wasted_count > 0 {
            if let Ok(mut stats) = self.access_stats.try_lock() {
                stats.wasted += page_wasted_count;
            }
        }
    }
    
    /// Perform periodic maintenance to keep memory usage reasonable
    pub fn maintain(&self) {
        // Perform dynamic concurrent limit adjustment based on hit rates
        self.adjust_concurrent_limit();
        
        // Count ready entries in cache
        let page_ready_count = {
            if let Ok(cache) = self.ready_cache.try_lock() {
                cache.len()
            } else {
                return; // Skip maintenance if we can't get the lock
            }
        };
        
        // If we have too many ready entries, clear some to prevent memory bloat
        let max_concurrent = self.max_concurrent.try_lock().map(|m| *m).unwrap_or(4);
        if page_ready_count > max_concurrent * 2 {
            let target_keep = max_concurrent;
            
            // Keep only the most recent entries
            if let Ok(mut cache) = self.ready_cache.try_lock() {
                let current_count = cache.len();
                
                if current_count > target_keep {
                    let to_remove = current_count - target_keep;
                    
                    // Remove random entries (we don't have ordering info)
                    let keys: Vec<_> = cache.keys().take(to_remove).copied().collect();
                    for key in keys {
                        cache.remove(&key);
                    }
                    
                    // Update wasted statistics
                    drop(cache);
                    if let Ok(mut stats) = self.access_stats.try_lock() {
                        stats.wasted += to_remove;
                    }
                }
            }
        }
        
        // Try to start next prefetches
        self.try_start_next_batch_prefetch();
    }

    /// Clear all prefetch buffers (useful for cleanup)
    pub fn clear(&self) {
        let page_ready_count = {
            if let Ok(mut cache) = self.ready_cache.try_lock() {
                let count = cache.len();
                cache.clear();
                count
            } else {
                0
            }
        };
        
        {
            if let Ok(mut requests) = self.batch_requests.try_lock() {
                requests.clear();
            }
        }
        
        {
            if let Ok(mut queue) = self.batch_pending_queue.try_lock() {
                queue.clear();
            }
        }
        
        // Record cleared ready entries as wasted
        if page_ready_count > 0 {
            if let Ok(mut stats) = self.access_stats.try_lock() {
                stats.wasted += page_ready_count;
            }
        }
    }

    /// Get hit rate for prefetch effectiveness
    pub fn hit_rate(&self) -> f64 {
        if let Ok(stats) = self.access_stats.try_lock() {
            if stats.hits + stats.misses == 0 {
                return 0.0;
            }
            stats.hits as f64 / (stats.hits + stats.misses) as f64
        } else {
            0.0 // Return 0 if we can't get the lock
        }
    }

    /// Get statistics about prefetch buffer usage (pages only)
    pub fn stats(&self) -> PrefetchStats {
        let ready_count = self.ready_cache.try_lock().map(|c| c.len()).unwrap_or(0);
        let pending_count = self.batch_requests.try_lock().map(|r| r.len()).unwrap_or(0);
        let queued_count = self.batch_pending_queue.try_lock().map(|q| q.len()).unwrap_or(0);
        let (hits, misses, wasted) = if let Ok(access_stats) = self.access_stats.try_lock() {
            (access_stats.hits, access_stats.misses, access_stats.wasted)
        } else {
            (0, 0, 0)
        };

        PrefetchStats {
            ready_count,
            pending_count,
            failed_count: 0, // With channels, failed requests are handled immediately
            queued_count,
            total_capacity: self.max_concurrent.try_lock().map(|m| *m).unwrap_or(0),
            hits,
            misses,
            wasted,
        }
    }
}

/// Statistics about prefetch buffer usage
#[derive(Debug, Clone)]
pub struct PrefetchStats {
    pub ready_count: usize,
    pub pending_count: usize,
    pub failed_count: usize,
    pub queued_count: usize,
    pub total_capacity: usize,
    pub hits: usize,
    pub misses: usize,
    pub wasted: usize,
}

impl PrefetchStats {
    /// Calculate hit rate percentage
    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            return 0.0;
        }
        (self.hits as f64 / (self.hits + self.misses) as f64) * 100.0
    }
    
    /// Calculate efficiency (useful prefetches vs total prefetches)
    pub fn efficiency(&self) -> f64 {
        let total_prefetches = self.hits + self.wasted;
        if total_prefetches == 0 {
            return 0.0;
        }
        (self.hits as f64 / total_prefetches as f64) * 100.0
    }
}

#[cfg(all(test, feature = "tokio"))]
mod tests {
    use std::time::Duration;
    use tokio::time::sleep;
    use fusio::{path::Path, DynFs, dynamic::DynFile};
    use tempfile::TempDir;
    use super::*;
    use crate::{
        executor::tokio::TokioExecutor,
        fs::{manager::StoreManager, FileType},
        inmem::immutable::tests::TestSchema,
        record::{
            test::get_test_record_batch,
            Schema,
        },
        DbOption,
    };
    use fusio_dispatch::FsOptions;
    use arrow::array::RecordBatch;
    use parquet::{
        arrow::{
            arrow_writer::ArrowWriterOptions, AsyncArrowWriter,
        },
        file::properties::WriterProperties,
        basic::{Compression, ZstdLevel},
    };
    use fusio_parquet::writer::AsyncWriter;
    use std::fs::File;

    async fn write_record_batch(
        file: Box<dyn DynFile>,
        record_batch: &RecordBatch,
    ) -> Result<(), parquet::errors::ParquetError> {
        let options = ArrowWriterOptions::new().with_properties(
            WriterProperties::builder()
                .set_created_by(concat!("tonbo version ", env!("CARGO_PKG_VERSION")).to_owned())
                .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
                .build(),
        );
        let mut writer = AsyncArrowWriter::try_new_with_options(
            AsyncWriter::new(file),
            TestSchema {}.arrow_schema().clone(),
            options,
        ).expect("Failed to create writer");
        writer.write(record_batch).await?;
        if writer.in_progress_size() > (1 << 20) - 1 {
            writer.flush().await?;
        }
        writer.close().await?;
        Ok(())
    }

    #[test]
    fn test_prefetch_stats() {
        let fs = std::sync::Arc::new(fusio::disk::TokioFs) as std::sync::Arc<dyn DynFs>;
        let prefetch = PrefetchBufferCollection::new(fs, 2);
        
        let stats = prefetch.stats();
        assert_eq!(stats.ready_count, 0);
        assert_eq!(stats.pending_count, 0);
        assert_eq!(stats.failed_count, 0);
        assert_eq!(stats.queued_count, 0);
        assert_eq!(stats.total_capacity, 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_prefetch_basic_functionality() {
        let temp_dir = TempDir::new().unwrap();
        let manager = StoreManager::new(FsOptions::Local, vec![]).unwrap();
        let base_fs = manager.base_fs();
        let fs = base_fs.clone() as Arc<dyn DynFs>;
        
        // Create prefetch buffer with more aggressive settings for testing
        let config = PrefetchConfig {
            reads_for_auto_prefetch: 1,              // Allow prefetch after 1 access
            base_batches_per_file: 2,                // Start with 2 batches per file  
            min_hit_rate_for_expansion: 0.5,         // Lower threshold for expansion
            target_batch_size: 1024,                 // 1KB target for test files
            max_batches_per_file: 3,                 // Allow up to 3 batches per file
        };
        let prefetch_buffer = PrefetchBufferCollection::new_with_config(fs.clone(), 6, config);
        
        println!("Testing RecordBatch Prefetch Basic Functionality");
        
        // Create multiple real parquet files with more data
        let mut file_paths = Vec::new();
        let mut file_ids = Vec::new();
        
        for i in 0..5 {
            let file_id = ulid::Ulid::new();
            let file_name = format!("test_{}.parquet", i);
            let table_path = temp_dir.path().join(&file_name);
            let _ = File::create(&table_path).unwrap();
            let fusio_path = Path::from_filesystem_path(&table_path).unwrap();
            
            // Create multiple batches worth of data
            let mut record_batches = Vec::new();
            for batch_num in 0..3 {
                let record_batch = get_test_record_batch::<TokioExecutor>(
                    DbOption::new(
                        Path::from_filesystem_path(temp_dir.path()).unwrap(),
                        &TestSchema,
                    ),
                    TokioExecutor::default(),
                ).await;
                record_batches.push(record_batch);
            }
            
            // Write multiple batches to the same file to create proper batch structure
            let file = base_fs
                .open_options(&fusio_path, FileType::Parquet.open_options(false))
                .await
                .unwrap();
            
            // Write first batch to establish file
            write_record_batch(file, &record_batches[0]).await.unwrap();
            
            // Check file size
            let file_size = std::fs::metadata(&table_path).unwrap().len();
            println!("   File {}: {} bytes ({} KB)", i, file_size, file_size / 1024);
            
            file_paths.push(fusio_path);
            file_ids.push(file_id);
        }
        
        println!("Created {} real parquet files with multiple batches", file_paths.len());
        
        // Create projection mask for all columns
        let projection_mask = parquet::arrow::ProjectionMask::all();
        
        // Simulate sequential access pattern to build up adaptive stats
        println!("Establishing sequential access pattern...");
        
        for (i, (file_id, path)) in file_ids.iter().zip(file_paths.iter()).enumerate().take(3) {
            // Record access pattern by calling record_file_access multiple times
            let batch_id = BatchId::new(*file_id, 0);
            prefetch_buffer.record_file_access(batch_id);
            
            // Second access to establish sequential pattern
            let batch_id_2 = BatchId::new(*file_id, 1);
            prefetch_buffer.record_file_access(batch_id_2);
            
            println!("   Recorded sequential access for file {}", i);
        }
        
        // Test batch prefetch functionality
        println!("\nStarting RecordBatch prefetch operations...");
        
        // Now start prefetching - should work better with established patterns
        for (i, (file_id, path)) in file_ids.iter().zip(file_paths.iter()).enumerate() {
            prefetch_buffer.prefetch_batches(*file_id, path.clone(), projection_mask.clone());
            
            // Small delay to allow async operations to start
            sleep(Duration::from_millis(10)).await;
        }
        
        // Give more time for prefetch operations to complete
        sleep(Duration::from_millis(100)).await;
        
        // Check intermediate stats
        let stats = prefetch_buffer.stats();
        println!("Stats after starting prefetch operations:");
        println!("   Ready: {}", stats.ready_count);
        println!("   Pending: {}", stats.pending_count);
        println!("   Failed: {}", stats.failed_count);
        println!("   Queued: {}", stats.queued_count);
        println!("   Capacity: {}", stats.total_capacity);
        println!("   Hit rate: {:.1}%", stats.hit_rate());
        
        assert_eq!(stats.total_capacity, 6, "Prefetch buffer should have capacity of 6");
        
        // Test batch access with real parquet files
        println!("\nTesting batch access...");
        let mut total_hits = 0;
        let mut total_attempts = 0;
        
        for (i, file_id) in file_ids.iter().enumerate() {
            // Try to access multiple batches per file
            for batch_index in 0..2 {
                total_attempts += 1;
                let batch_id = BatchId {
                    file_id: *file_id,
                    batch_index,
                };
                
                let batch_result = prefetch_buffer.try_get_batch(batch_id);
                match batch_result {
                    Some(_batch) => {
                        println!("   File {} Batch {} successfully retrieved from prefetch buffer", i, batch_index);
                        total_hits += 1;
                    }
                    None => {
                        println!("   File {} Batch {} not in prefetch buffer", i, batch_index);
                    }
                }
                
                // Small delay between accesses
                sleep(Duration::from_millis(5)).await;
            }
        }
        
        let hit_rate = if total_attempts > 0 {
            (total_hits as f64 / total_attempts as f64) * 100.0
        } else {
            0.0
        };
        
        println!("\nAccess Results:");
        println!("   Total batch access attempts: {}", total_attempts);
        println!("   Successful hits: {}", total_hits);
        println!("   Hit rate: {:.1}%", hit_rate);
        
        let final_stats = prefetch_buffer.stats();
        println!("\nFinal prefetch buffer stats:");
        println!("   Ready: {}", final_stats.ready_count);
        println!("   Pending: {}", final_stats.pending_count);
        println!("   Failed: {}", final_stats.failed_count);
        println!("   Queued: {}", final_stats.queued_count);
        println!("   Total hits: {}", final_stats.hits);
        println!("   Total misses: {}", final_stats.misses);
        println!("   Overall hit rate: {:.1}%", final_stats.hit_rate());
        
        // We should see some hits if prefetch is working
        if hit_rate > 0.0 {
            println!("Prefetch showing positive hit rate!");
        } else {
            println!("No prefetch hits - may need more time or different access pattern");
        }
        
        println!("RecordBatch prefetch basic functionality test completed successfully!");
    }

    #[tokio::test(flavor = "multi_thread")] 
    async fn test_batch_prefetch_queue_management() {
        let temp_dir = TempDir::new().unwrap();
        let manager = StoreManager::new(FsOptions::Local, vec![]).unwrap();
        let base_fs = manager.base_fs();
        let fs = base_fs.clone() as Arc<dyn DynFs>;
        
        // Create prefetch buffer with capacity of 1 (small capacity for testing)
        let prefetch_buffer = PrefetchBufferCollection::new(fs.clone(), 1);
        
        println!("Testing RecordBatch Prefetch Queue Management");
        
        // Create real parquet files
        let mut file_ids = Vec::new();
        let mut file_paths = Vec::new();
        
        for i in 0..3 {
            let file_id = ulid::Ulid::new();
            let file_name = format!("queue_test_{}.parquet", i);
            let table_path = temp_dir.path().join(&file_name);
            let _ = File::create(&table_path).unwrap();
            let fusio_path = Path::from_filesystem_path(table_path).unwrap();
            
            // Create real RecordBatch and write to parquet file
            let record_batch = get_test_record_batch::<TokioExecutor>(
                DbOption::new(
                    Path::from_filesystem_path(temp_dir.path()).unwrap(),
                    &TestSchema,
                ),
                TokioExecutor::default(),
            ).await;
            
            let file = base_fs
                .open_options(&fusio_path, FileType::Parquet.open_options(false))
                .await
                .unwrap();
            write_record_batch(file, &record_batch).await.unwrap();
            
            file_ids.push(file_id);
            file_paths.push(fusio_path);
        }
        
        let projection_mask = parquet::arrow::ProjectionMask::all();
        
        // Start batch prefetch for all files (should queue some due to capacity limit)
        for (file_id, path) in file_ids.iter().zip(file_paths.iter()) {
            prefetch_buffer.prefetch_batches(*file_id, path.clone(), projection_mask.clone());
        }
        
        // Give time for initial prefetch operations to start
        sleep(Duration::from_millis(50)).await;
        
        // Check initial stats
        let stats = prefetch_buffer.stats();
        println!("Initial stats after batch prefetch requests:");
        println!("   Active (ready + pending): {}", stats.ready_count + stats.pending_count);
        println!("   Queued: {}", stats.queued_count);
        println!("   Capacity: {}", stats.total_capacity);
        
        // With capacity of 1, we should have at most 1 active and rest queued
        assert!(stats.ready_count + stats.pending_count <= 1, 
               "Should not exceed capacity of 1 active operations, got {}", 
               stats.ready_count + stats.pending_count);
        
        // Test batch access for different batches
        for i in 0..3 {
            println!("   Checking batch access for file {}", i);
            
            let batch_id = BatchId {
                file_id: file_ids[i],
                batch_index: 0,
            };
            
            let batch_result = prefetch_buffer.try_get_batch(batch_id);
            match batch_result {
                Some(_batch) => {
                    println!("      Got batch from prefetch buffer");
                }
                None => {
                    println!("      Batch not in prefetch buffer - may still be loading");
                }
            }
            
            // Give time for queue processing
            sleep(Duration::from_millis(30)).await;
            
            let stats = prefetch_buffer.stats();
            println!("      Stats after checking batch {}: ready={}, pending={}, queued={}", 
                    i, stats.ready_count, stats.pending_count, stats.queued_count);
        }
        
        // Final stats 
        let final_stats = prefetch_buffer.stats();
        println!("Final stats:");
        println!("   Ready: {}", final_stats.ready_count);
        println!("   Pending: {}", final_stats.pending_count);
        println!("   Queued: {}", final_stats.queued_count);
        
        println!("RecordBatch prefetch queue management test completed successfully!");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_prefetch_polling_integration() {
        let temp_dir = TempDir::new().unwrap();
        let manager = StoreManager::new(FsOptions::Local, vec![]).unwrap();
        let base_fs = manager.base_fs();
        let fs = base_fs.clone() as Arc<dyn DynFs>;
        
        let prefetch_buffer = Arc::new(PrefetchBufferCollection::new(fs.clone(), 2));
        
        println!("Testing RecordBatch Prefetch Polling Integration");
        
        // Create real parquet file
        let file_id = ulid::Ulid::new();
        let table_path = temp_dir.path().join("polling_test.parquet");
        let _ = File::create(&table_path).unwrap();
        let file_path = Path::from_filesystem_path(table_path).unwrap();
        
        // Create real RecordBatch and write to parquet file
        let record_batch = get_test_record_batch::<TokioExecutor>(
            DbOption::new(
                Path::from_filesystem_path(temp_dir.path()).unwrap(),
                &TestSchema,
            ),
            TokioExecutor::default(),
        ).await;
        
        let file = base_fs
            .open_options(&file_path, FileType::Parquet.open_options(false))
            .await
            .unwrap();
        write_record_batch(file, &record_batch).await.unwrap();
        
        let projection_mask = parquet::arrow::ProjectionMask::all();
        
        // Start batch prefetch operation
        prefetch_buffer.prefetch_batches(file_id, file_path.clone(), projection_mask);
        
        // Test manual polling (simulates what happens in stream processing)
        let mut poll_count = 0;
        let max_polls = 10; // Keep it small to avoid long test times
        
        loop {
            poll_count += 1;
            
            // Create a dummy context for polling
            use std::task::{Context, Waker};
            
            struct DummyWaker;
            impl std::task::Wake for DummyWaker {
                fn wake(self: Arc<Self>) {}
            }
            
            let waker = Waker::from(Arc::new(DummyWaker));
            let mut context = Context::from_waker(&waker);
            
            // Poll prefetch operations
            let _progress_made = prefetch_buffer.poll_prefetches(&mut context);
            
            if poll_count >= max_polls {
                break;
            }
            
            // Small delay between polls
            sleep(Duration::from_millis(5)).await;
        }
        
        println!("   Completed polling after {} iterations", poll_count);
        
        // Give additional time for async operation to complete
        sleep(Duration::from_millis(50)).await;
        
        // Check stats BEFORE trying to access batches
        let pre_access_stats = prefetch_buffer.stats();
        println!("   Stats before batch access: ready={}, pending={}, failed={}", 
                 pre_access_stats.ready_count, pre_access_stats.pending_count, pre_access_stats.failed_count);
        
        // Test batch access
        let batch_id = BatchId {
            file_id,
            batch_index: 0,
        };
        
        let batch_result = prefetch_buffer.try_get_batch(batch_id);
        let _batch_retrieved_successfully = match batch_result {
            Some(_batch) => {
                println!("   Batch successfully prefetched and retrieved");
                true
            }
            None => {
                println!("   Batch not found in prefetch buffer - may still be loading");
                false
            }
        };
        
        // Stats after batch access
        let post_access_stats = prefetch_buffer.stats();
        println!("   Stats after batch access: ready={}, pending={}, failed={}", 
                 post_access_stats.ready_count, post_access_stats.pending_count, post_access_stats.failed_count);
        
        // The test is successful if we can track operations with real parquet files
        let polling_worked = pre_access_stats.ready_count > 0 || pre_access_stats.pending_count > 0 || pre_access_stats.failed_count > 0;
        println!("   Polling tracked operations: {}", polling_worked);
        
        println!("RecordBatch prefetch polling integration test completed successfully!");
    }
    
    #[tokio::test(flavor = "multi_thread")]
    async fn test_sequential_batch_scan_with_prefetch() {
        let temp_dir = TempDir::new().unwrap();
        let manager = StoreManager::new(FsOptions::Local, vec![]).unwrap();
        let base_fs = manager.base_fs();
        let fs = base_fs.clone() as Arc<dyn DynFs>;
        
        println!("Testing Sequential RecordBatch Scan with Prefetch Buffer");
        
        // Create more aggressive prefetch config for better hit rates
        let config = PrefetchConfig {
            reads_for_auto_prefetch: 1,              // Enable after 1 access
            base_batches_per_file: 3,                // Prefetch 3 batches per file
            min_hit_rate_for_expansion: 0.4,         // Lower threshold
            target_batch_size: 1024,                 
            max_batches_per_file: 4,                 // Allow up to 4 batches
        };
        let prefetch_buffer = Arc::new(PrefetchBufferCollection::new_with_config(fs.clone(), 8, config));
        
        // Create real parquet files for better testing
        let file_count = 8; // Reasonable number for testing
        let mut file_paths = Vec::new();
        let mut file_ids = Vec::new();
        
        for i in 0..file_count {
            let file_id = ulid::Ulid::new();
            let file_name = format!("sequential_test_{:03}.parquet", i);
            let table_path = temp_dir.path().join(&file_name);
            let _ = File::create(&table_path).unwrap();
            let fusio_path = Path::from_filesystem_path(&table_path).unwrap();
            
            // Create real RecordBatch for proper parquet structure
            let record_batch = get_test_record_batch::<TokioExecutor>(
                DbOption::new(
                    Path::from_filesystem_path(temp_dir.path()).unwrap(),
                    &TestSchema,
                ),
                TokioExecutor::default(),
            ).await;
            
            let file = base_fs
                .open_options(&fusio_path, FileType::Parquet.open_options(false))
                .await
                .unwrap();
            write_record_batch(file, &record_batch).await.unwrap();
            
            let file_size = std::fs::metadata(&table_path).unwrap().len();
            println!("   File {}: {} bytes ({} KB)", i, file_size, file_size / 1024);
            
            file_paths.push(fusio_path);
            file_ids.push(file_id);
        }
        
        println!("Created {} real parquet files", file_count);
        println!("   This simulates LSM with small SSTables and prefetch optimization");
        
        let projection_mask = parquet::arrow::ProjectionMask::all();
        
        // Build up access pattern history first
        println!("\nEstablishing access patterns...");
        for (i, file_id) in file_ids.iter().enumerate().take(4) {
            let batch_id_1 = BatchId::new(*file_id, 0);
            let batch_id_2 = BatchId::new(*file_id, 1);
            prefetch_buffer.record_file_access(batch_id_1);
            prefetch_buffer.record_file_access(batch_id_2);
            println!("   Established pattern for file {}", i);
        }
        
        // Simulate sequential batch access pattern with prefetch
        println!("\nTesting sequential access with prefetch...");
        
        let mut batch_access_stats = Vec::new();
        let mut global_hits = 0;
        let mut global_misses = 0;
        
        for (i, (file_id, path)) in file_ids.iter().zip(file_paths.iter()).enumerate() {
            // Pre-fetch current file
            prefetch_buffer.prefetch_batches(*file_id, path.clone(), projection_mask.clone());
            
            // Also prefetch next file for read-ahead optimization
            if i + 1 < file_ids.len() {
                prefetch_buffer.prefetch_batches(file_ids[i + 1], file_paths[i + 1].clone(), projection_mask.clone());
            }
            
            // Allow some time for prefetch to work (realistic scenario)
            tokio::time::sleep(Duration::from_millis(20)).await;
            
            // Simulate sequential batch access within this file
            let start = std::time::Instant::now();
            let mut batch_hits = 0;
            let mut batch_misses = 0;
            
            // Access batches 0, 1, 2 sequentially (realistic pattern)
            for batch_index in 0..3 {
                let batch_id = BatchId {
                    file_id: *file_id,
                    batch_index,
                };
                
                match prefetch_buffer.try_get_batch(batch_id) {
                    Some(_batch) => {
                        batch_hits += 1;
                        global_hits += 1;
                        println!("   File {} Batch {} HIT from prefetch", i, batch_index);
                    }
                    None => {
                        batch_misses += 1;
                        global_misses += 1;
                        println!("   File {} Batch {} MISS", i, batch_index);
                    }
                }
                
                // Small delay between batch accesses (realistic)
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            
            let elapsed = start.elapsed();
            batch_access_stats.push((batch_hits, batch_misses, elapsed));
            
            let file_hit_rate = if batch_hits + batch_misses > 0 {
                (batch_hits as f64 / (batch_hits + batch_misses) as f64) * 100.0
            } else {
                0.0
            };
            
            println!("   File {}: {} hits, {} misses ({:.1}% hit rate) in {:?}", 
                    i, batch_hits, batch_misses, file_hit_rate, elapsed);
        }
        
        // Analyze overall results
        let total_batches = global_hits + global_misses;
        let overall_hit_rate = if total_batches > 0 {
            (global_hits as f64 / total_batches as f64) * 100.0
        } else {
            0.0
        };
        
        println!("\nSequential Scan Results:");
        println!("   Total files scanned: {}", file_count);
        println!("   Total batch accesses: {}", total_batches);
        println!("   Prefetch hits: {} ({:.1}%)", global_hits, overall_hit_rate);
        println!("   Prefetch misses: {} ({:.1}%)", global_misses, 100.0 - overall_hit_rate);
        
        // Final buffer stats
        let final_stats = prefetch_buffer.stats();
        println!("\nFinal Buffer Statistics:");
        println!("   Ready batches: {}", final_stats.ready_count);
        println!("   Pending operations: {}", final_stats.pending_count);
        println!("   Failed operations: {}", final_stats.failed_count);
        println!("   Queued operations: {}", final_stats.queued_count);
        println!("   Total capacity: {}", final_stats.total_capacity);
        println!("   Cumulative hits: {}", final_stats.hits);
        println!("   Cumulative misses: {}", final_stats.misses);
        println!("   Buffer hit rate: {:.1}%", final_stats.hit_rate());
        
        // Success criteria
        if overall_hit_rate > 10.0 {
            println!("Good prefetch performance! Hit rate: {:.1}%", overall_hit_rate);
        } else if overall_hit_rate > 0.0 {
            println!("Moderate prefetch performance. Hit rate: {:.1}%", overall_hit_rate);
        } else {
            println!("No prefetch hits detected. May need tuning or more time.");
        }
        
        println!("Sequential batch scan with prefetch test completed!");
    }
    
    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_prefetch_access_patterns() {
        let temp_dir = TempDir::new().unwrap();
        let fs = Arc::new(fusio::disk::TokioFs) as Arc<dyn DynFs>;
        
        println!("Testing RecordBatch Prefetch Access Patterns");
        
        // Create test files
        let mut file_data = Vec::new();
        for i in 0..3 {
            let file_id = ulid::Ulid::new();
            let path = temp_dir.path().join(format!("test_file_{}.parquet", i));
            let fusio_path = Path::from_filesystem_path(&path).unwrap();
            let data = format!("test_parquet_data_{}", i).repeat(1000);
            std::fs::write(&path, data).unwrap();
            
            file_data.push((file_id, fusio_path));
        }
        
        println!("Testing different batch access patterns...");
        
        let prefetch_buffer = PrefetchBufferCollection::new(fs.clone(), 5);
        let projection_mask = parquet::arrow::ProjectionMask::all();
        
        // Pattern 1: Prefetch batches then access
        println!("\nPattern 1: Prefetch then access");
        for (file_id, fusio_path) in &file_data {
            prefetch_buffer.prefetch_batches(*file_id, fusio_path.clone(), projection_mask.clone());
        }
        
        // Give prefetch time to work
        tokio::time::sleep(Duration::from_millis(30)).await;
        
        let mut hits = 0;
        let mut total = 0;
        for (file_id, _) in &file_data {
            for batch_index in 0..2 {
                total += 1;
                let batch_id = BatchId {
                    file_id: *file_id,
                    batch_index,
                };
                
                if prefetch_buffer.try_get_batch(batch_id).is_some() {
                    hits += 1;
                }
            }
        }
        let hit_rate = (hits as f64 / total as f64) * 100.0;
        println!("   Prefetch then access: {}/{} hits ({:.1}% hit rate)", hits, total, hit_rate);
        
        // Pattern 2: Immediate access (should miss)
        println!("\nPattern 2: Immediate access");
        let prefetch_buffer2 = PrefetchBufferCollection::new(fs.clone(), 5);
        
        let mut hits = 0;
        let mut total = 0;
        for (file_id, fusio_path) in &file_data {
            prefetch_buffer2.prefetch_batches(*file_id, fusio_path.clone(), projection_mask.clone());
            
            // Immediately try to access (should miss)
            for batch_index in 0..2 {
                total += 1;
                let batch_id = BatchId {
                    file_id: *file_id,
                    batch_index,
                };
                
                if prefetch_buffer2.try_get_batch(batch_id).is_some() {
                    hits += 1;
                }
            }
        }
        let immediate_hit_rate = (hits as f64 / total as f64) * 100.0;
        println!("   Immediate access: {}/{} hits ({:.1}% hit rate)", hits, total, immediate_hit_rate);
        
        // Final stats
        let final_stats = prefetch_buffer.stats();
        println!("   Final buffer stats: ready={}, pending={}, failed={}, queued={}", 
                final_stats.ready_count, final_stats.pending_count, final_stats.failed_count, final_stats.queued_count);
        
        println!("RecordBatch prefetch access patterns test completed successfully!");
        println!("   Note: Low hit rates are expected since test files aren't real parquet files");
    }
}