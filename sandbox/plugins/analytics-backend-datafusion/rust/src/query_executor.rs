/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

use std::sync::Arc;

use datafusion::{
    common::DataFusionError,
    datasource::listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    execution::context::SessionContext,
    execution::runtime_env::RuntimeEnvBuilder,
    execution::SessionStateBuilder,
    physical_plan::execute_stream,
    prelude::*,
};
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::execution::cache::cache_manager::CacheManagerConfig;
use datafusion::execution::cache::{CacheAccessor, DefaultListFilesCache};
use datafusion_substrait::logical_plan::consumer::from_substrait_plan;
use log::error;
use object_store::ObjectMeta;
use prost::Message;
use substrait::proto::Plan;

use crate::cross_rt_stream::CrossRtStream;
use crate::executor::DedicatedExecutor;
use crate::api::DataFusionRuntime;

/// Execute a vanilla parquet query: substrait plan → DataFusion → CrossRtStream.
/// File access goes through DataFusion's registered object store.
pub async fn execute_query(
    table_path: ListingTableUrl,
    object_metas: Arc<Vec<ObjectMeta>>,
    table_name: String,
    plan_bytes: Vec<u8>,
    runtime: &DataFusionRuntime,
    cpu_executor: DedicatedExecutor,
    // Per-query memory pool, or None when context_id is 0 (tracking disabled).
    // Not all query flows pass a context_id yet; this fallback allows queries
    // to execute using the global pool. Can be made required once all flows
    // wire up context_id correctly.
    query_memory_pool: Option<Arc<dyn datafusion::execution::memory_pool::MemoryPool>>,
) -> Result<i64, DataFusionError> {
    // Pre-populate the list-files cache so DataFusion doesn't re-list the directory
    let list_file_cache = Arc::new(DefaultListFilesCache::default());
    let table_scoped_path = datafusion::execution::cache::TableScopedPath {
        table: None,
        path: table_path.prefix().clone(),
    };
    list_file_cache.put(&table_scoped_path, object_metas);

    // Build a per-query RuntimeEnv sharing the global memory pool + caches,
    // but with a fresh list-files cache for this query's shard files.
    let mut runtime_env_builder = RuntimeEnvBuilder::from_runtime_env(&runtime.runtime_env)
        .with_cache_manager(
            CacheManagerConfig::default()
                .with_list_files_cache(Some(list_file_cache))
                .with_file_metadata_cache(Some(
                    runtime.runtime_env.cache_manager.get_file_metadata_cache(),
                ))
                .with_files_statistics_cache(
                    runtime.runtime_env.cache_manager.get_file_statistic_cache(),
                ),
        );

    // If a per-query memory pool is provided, set it on the same builder.
    // The per-query pool wraps the global pool, so global limits are still enforced.
    if let Some(pool) = query_memory_pool {
        runtime_env_builder = runtime_env_builder.with_memory_pool(pool);
    }

    let runtime_env = runtime_env_builder
        .build()
        .map_err(|e| {
            error!("Failed to build runtime env: {}", e);
            e
        })?;

    // Build a fresh session state per query. TODO : Tune this during planning per query
    let mut config = SessionConfig::new();
    config.options_mut().execution.parquet.pushdown_filters = false;
    config.options_mut().execution.target_partitions = 16;
    config.options_mut().execution.batch_size = 8192;

    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(Arc::from(runtime_env))
        .with_default_features()
        .build();

    let ctx = SessionContext::new_with_state(state);

    // Register table via ListingTable — all IO goes through object store
    let file_format = ParquetFormat::new();
    let listing_options = ListingOptions::new(Arc::new(file_format))
        .with_file_extension(".parquet")
        .with_collect_stat(true);

    let resolved_schema = listing_options
        .infer_schema(&ctx.state(), &table_path)
        .await
        .map_err(|e| {
            error!("Failed to infer schema: {}", e);
            e
        })?;

    let table_config = ListingTableConfig::new(table_path)
        .with_listing_options(listing_options)
        .with_schema(resolved_schema);

    let provider = Arc::new(ListingTable::try_new(table_config).map_err(|e| {
        error!("Failed to create listing table: {}", e);
        e
    })?);

    ctx.register_table(&table_name, provider).map_err(|e| {
        error!("Failed to register table: {}", e);
        e
    })?;

    // Decode substrait → logical plan → physical plan → stream
    let substrait_plan = Plan::decode(plan_bytes.as_slice()).map_err(|e| {
        DataFusionError::Execution(format!("Failed to decode Substrait: {}", e))
    })?;

    let logical_plan = from_substrait_plan(&ctx.state(), &substrait_plan).await?;
    let dataframe = ctx.execute_logical_plan(logical_plan).await?;
    let physical_plan = dataframe.create_physical_plan().await?;

    log::info!(
        "[df_execute_query] physical_plan partitions={} plan={}",
        physical_plan.properties().output_partitioning().partition_count(),
        datafusion::physical_plan::displayable(physical_plan.as_ref()).indent(false)
    );

    let df_stream = execute_stream(physical_plan, ctx.task_ctx()).map_err(|e| {
        error!("Failed to create execution stream: {}", e);
        e
    })?;

    // Wrap in CrossRtStream — CPU work runs on DedicatedExecutor
    let cross_rt_stream =
        CrossRtStream::new_with_df_error_stream(df_stream, cpu_executor);
    let wrapped = datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(
        cross_rt_stream.schema(),
        cross_rt_stream,
    );

    Ok(Box::into_raw(Box::new(wrapped)) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow_array::{Int64Array, RecordBatch};
    use datafusion::datasource::listing::ListingTableUrl;
    use datafusion::execution::disk_manager::DiskManagerBuilder;
    use datafusion::execution::memory_pool::GreedyMemoryPool;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_substrait::logical_plan::producer::to_substrait_plan;
    use futures::TryStreamExt;
    use object_store::local::LocalFileSystem;
    use object_store::ObjectStore;
    use parquet::arrow::ArrowWriter;
    use prost::Message;
    use std::fs::File;
    use std::sync::Arc;
    use crate::api::DataFusionRuntime;
    use crate::cross_rt_stream::CrossRtStream;
    use crate::runtime_manager::RuntimeManager;

    fn create_test_parquet(dir: &std::path::Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]));
        let ids: Vec<i64> = (0..1000).collect();
        let vals: Vec<i64> = ids.iter().map(|i| i * 10).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(ids)), Arc::new(Int64Array::from(vals))],
        ).unwrap();
        let path = dir.join("test.parquet");
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn test_execute_query_with_tracking_allocator() {
        // This test exercises the same path as the benchmark:
        // TrackingAllocator -> RuntimeManager -> execute_query -> DataFusion
        use native_bridge_common::allocator::register_plugin;
        use crate::ffm::PLUGIN_ID;
        PLUGIN_ID.get_or_init(|| register_plugin("datafusion"));
        let mgr = RuntimeManager::new(4);
        let runtime_env = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(256 * 1024 * 1024)))
            .with_disk_manager_builder(DiskManagerBuilder::default())
            .build().unwrap();
        let df_runtime = DataFusionRuntime { runtime_env };
        let tmp = tempfile::tempdir().unwrap();
        create_test_parquet(tmp.path());

        let dir = tmp.path().to_str().unwrap();
        let url = ListingTableUrl::parse(dir).unwrap();

        // Get object metas
        let store = Arc::new(LocalFileSystem::new());
        let path = object_store::path::Path::from(format!("{}/test.parquet", dir));
        let meta = mgr.io_runtime.block_on(store.head(&path)).unwrap();
        let metas = Arc::new(vec![meta]);

        // Generate substrait plan
        let plan_bytes = mgr.io_runtime.block_on(async {
            let ctx = datafusion::prelude::SessionContext::new();
            let opts = datafusion::datasource::listing::ListingOptions::new(
                Arc::new(datafusion::datasource::file_format::parquet::ParquetFormat::new())
            ).with_file_extension(".parquet").with_collect_stat(true);
            let schema = opts.infer_schema(&ctx.state(), &url).await.unwrap();
            let cfg = datafusion::datasource::listing::ListingTableConfig::new(url.clone())
                .with_listing_options(opts).with_schema(schema);
            ctx.register_table("t", Arc::new(datafusion::datasource::listing::ListingTable::try_new(cfg).unwrap())).unwrap();
            let plan = ctx.sql("SELECT SUM(value), COUNT(*) FROM t").await.unwrap().logical_plan().clone();
            let sub = to_substrait_plan(&plan, &ctx.state()).unwrap();
            let mut buf = Vec::new();
            sub.encode(&mut buf).unwrap();
            buf
        });

        // Execute query (same path as df_execute_query FFI)
        let result = mgr.io_runtime.block_on(async {
            let exec = mgr.cpu_executor();
            execute_query(
                url, metas, "t".into(), plan_bytes, &df_runtime, exec, None,
            ).await
        });

        assert!(result.is_ok(), "Query failed: {:?}", result.err());

        // Consume the stream
        let ptr = result.unwrap();
        mgr.io_runtime.block_on(async {
            let mut stream = unsafe {
                Box::from_raw(ptr as *mut datafusion::physical_plan::stream::RecordBatchStreamAdapter<CrossRtStream>)
            };
            let mut rows = 0u64;
            while let Some(batch) = stream.try_next().await.unwrap() {
                rows += batch.num_rows() as u64;
            }
            assert!(rows > 0, "Expected at least one row");
        });
    }
}
