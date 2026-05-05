/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// Single cdylib for JDK FFM (Foreign Function & Memory API).
//
// This crate:
//   1. Sets the global jemalloc allocator (shared across all plugin rlibs)
//   2. Pulls in plugin rlibs via extern crate (forces linker to include symbols)
//   3. All #[no_mangle] extern "C" functions from the plugin crates are
//      automatically available for dlsym/SymbolLookup
// ═══════════════════════════════════════════════════════════════════════════════

use std::sync::atomic::{AtomicBool, Ordering};
use tikv_jemalloc_ctl::{epoch, stats};

#[export_name = "malloc_conf"]
pub static MALLOC_CONF: &[u8] = b"dirty_decay_ms:30000,muzzy_decay_ms:30000,lg_tcache_max:16\0";

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Pull in plugin rlibs — forces linker to include all #[no_mangle] symbols.
extern crate native_bridge_common;
extern crate opensearch_datafusion;
extern crate opensearch_parquet_format;
extern crate opensearch_repository_s3;
extern crate opensearch_repository_gcs;
extern crate opensearch_repository_azure;
extern crate opensearch_repository_fs;
extern crate opensearch_tiered_storage;

// ── Periodic jemalloc metrics logging ───────────────────────────────────────

static METRICS_STARTED: AtomicBool = AtomicBool::new(false);

/// Start periodic jemalloc metrics logging at the given interval (seconds).
/// Safe to call multiple times — only the first call spawns the thread.
/// Log format matches mimalloc branch for CW Logs Insights compatibility.
#[no_mangle]
pub extern "C" fn native_mimalloc_metrics_start(interval_secs: i64) {
    if METRICS_STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    let interval = std::time::Duration::from_secs(interval_secs.max(1) as u64);
    std::thread::Builder::new()
        .name("jemalloc-metrics".into())
        .spawn(move || {
            let mut peak_allocated: u64 = 0;
            let mut peak_resident: u64 = 0;
            loop {
                std::thread::sleep(interval);
                // Advance epoch to flush thread-cached stats
                if epoch::advance().is_err() {
                    continue;
                }
                let allocated = stats::allocated::read().unwrap_or(0) as u64;
                let resident = stats::resident::read().unwrap_or(0) as u64;
                peak_allocated = peak_allocated.max(allocated);
                peak_resident = peak_resident.max(resident);
                let mb = |b: u64| b / (1024 * 1024);
                native_bridge_common::log_info!(
                    "[mimalloc-metrics] allocated={}MB peak_allocated={}MB resident={}MB peak_resident={}MB",
                    mb(allocated), mb(peak_allocated), mb(resident), mb(peak_resident),
                );
            }
        })
        .expect("Failed to spawn jemalloc-metrics thread");
}
