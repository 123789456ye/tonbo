use std::{
    marker::PhantomData,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use arrow::datatypes::Schema;
use futures_core::{ready, Stream};
use parquet::arrow::{
    async_reader::{AsyncFileReader, ParquetRecordBatchStream},
    ProjectionMask,
};
use pin_project_lite::pin_project;

use crate::{
    fs::FileId,
    option::Order,
    record::Record,
    stream::{
        prefetch::{PrefetchBufferCollection, BatchId},
        record_batch::{RecordBatchEntry, RecordBatchIterator},
    },
};

pin_project! {
    pub struct SsTableScan<'scan, R> {
        #[pin]
        stream: ParquetRecordBatchStream<Box<dyn AsyncFileReader>>,
        iter: Option<RecordBatchIterator<R>>,
        projection_mask: ProjectionMask,
        full_schema: Arc<Schema>,
        order: Option<Order>,
        // Prefetch support
        file_id: Option<FileId>,
        prefetch_buffer: Option<std::sync::Arc<PrefetchBufferCollection>>,
        current_batch_index: usize,
        _marker: PhantomData<&'scan ()>
    }
}

impl<R> SsTableScan<'_, R> {
    pub fn new(
        stream: ParquetRecordBatchStream<Box<dyn AsyncFileReader>>,
        projection_mask: ProjectionMask,
        full_schema: Arc<Schema>,
        order: Option<Order>,
    ) -> Self {
        SsTableScan {
            stream,
            iter: None,
            projection_mask,
            full_schema,
            order,
            file_id: None,
            prefetch_buffer: None,
            current_batch_index: 0,
            _marker: PhantomData,
        }
    }
    
    /// Create a new SsTableScan with prefetch support
    pub fn new_with_prefetch(
        stream: ParquetRecordBatchStream<Box<dyn AsyncFileReader>>,
        projection_mask: ProjectionMask,
        full_schema: Arc<Schema>,
        order: Option<Order>,
        file_id: FileId,
        prefetch_buffer: Arc<PrefetchBufferCollection>,
    ) -> Self {
        SsTableScan {
            stream,
            iter: None,
            projection_mask,
            full_schema,
            order,
            file_id: Some(file_id),
            prefetch_buffer: Some(prefetch_buffer),
            current_batch_index: 0,
            _marker: PhantomData,
        }
    }
}

impl<R> Stream for SsTableScan<'_, R>
where
    R: Record,
{
    type Item = Result<RecordBatchEntry<R>, parquet::errors::ParquetError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        loop {
            match this.iter {
                Some(iter) => {
                    if let Some(entry) = iter.next() {
                        return Poll::Ready(Some(Ok(entry)));
                    }
                    *this.iter = None;
                }
                None => {
                    // Extract values to avoid borrow conflicts  
                    let file_id = *this.file_id;
                    let current_batch_index = *this.current_batch_index;
                    
                    // Try to get batch from prefetch buffer first if available
                    if let (Some(file_id), Some(prefetch_buffer)) = (file_id, this.prefetch_buffer.as_ref()) {
                        // Poll prefetch operations to advance their state
                        prefetch_buffer.poll_prefetches(cx);
                        
                        let batch_id = BatchId::new(file_id, current_batch_index);
                        if let Some(batch_data) = prefetch_buffer.try_get_batch(batch_id) {
                            // Got batch from prefetch buffer - need to keep stream in sync
                            // We consume one batch from the stream to keep indices aligned
                            let _stream_batch = ready!(this.stream.as_mut().poll_next(cx)).transpose()?;
                            
                            // Check if stream is exhausted
                            if _stream_batch.is_none() {
                                return Poll::Ready(None);
                            }
                            
                            // Use the prefetched batch instead of the stream batch
                            *this.current_batch_index += 1;
                            
                            *this.iter = Some(RecordBatchIterator::new(
                                batch_data.batch,
                                this.projection_mask.clone(),
                                this.full_schema.clone(),
                                *this.order,
                            ));
                            continue;
                        }
                    }
                    
                    // Fall back to reading from stream if prefetch miss or no prefetch buffer
                    let record_batch = ready!(this.stream.as_mut().poll_next(cx)).transpose()?;
                    
                    let record_batch = match record_batch {
                        Some(record_batch) => record_batch,
                        None => return Poll::Ready(None),
                    };
                    
                    // Increment batch index for prefetch coordination tracking
                    if file_id.is_some() {
                        *this.current_batch_index += 1;
                    }
                    
                    *this.iter = Some(RecordBatchIterator::new(
                        record_batch,
                        this.projection_mask.clone(),
                        this.full_schema.clone(),
                        *this.order,
                    ));
                }
            }
        }
    }
}
