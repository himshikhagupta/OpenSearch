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
// Unlike the JNI approach (RegisterNatives, classloader workarounds), FFM calls
// extern "C" functions directly via SymbolLookup + Linker.downcallHandle().
// No JNIEnv, no JClass, no classloader binding — just plain C ABI.
//
// This crate:
//   1. Sets the global mimalloc allocator (shared across all plugin rlibs)
//   2. Pulls in plugin rlibs via extern crate (forces linker to include symbols)
//   3. All #[no_mangle] extern "C" functions from the plugin crates are
//      automatically available for dlsym/SymbolLookup
// ═══════════════════════════════════════════════════════════════════════════════

use core::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicBool, Ordering};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Pull in plugin rlibs — forces linker to include all #[no_mangle] symbols.
extern crate native_bridge_common;
extern crate opensearch_datafusion;
extern crate opensearch_parquet_format;
extern crate opensearch_repository_s3;
extern crate opensearch_repository_gcs;
extern crate opensearch_repository_azure;
extern crate opensearch_repository_fs;
extern crate opensearch_tiered_storage;

// ── Periodic mimalloc metrics logging ───────────────────────────────────────

extern "C" {
    fn mi_stats_get_json(buf_size: usize, buf: *mut c_char) -> *mut c_char;
}

static METRICS_STARTED: AtomicBool = AtomicBool::new(false);

/// Start periodic mimalloc metrics logging at the given interval (seconds).
/// Safe to call multiple times — only the first call spawns the thread.
#[no_mangle]
pub extern "C" fn native_mimalloc_metrics_start(interval_secs: i64) {
    if METRICS_STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    let interval = std::time::Duration::from_secs(interval_secs.max(1) as u64);
    std::thread::Builder::new()
        .name("mimalloc-metrics".into())
        .spawn(move || loop {
            std::thread::sleep(interval);
            let (current, peak) = committed_stats();
            native_bridge_common::log_info!(
                "[mimalloc-metrics] committed={}MB peak_committed={}MB",
                current / (1024 * 1024),
                peak / (1024 * 1024),
            );
        })
        .expect("Failed to spawn mimalloc-metrics thread");
}

/// Returns (committed_current, committed_peak) from mimalloc's internal stats.
fn committed_stats() -> (i64, i64) {
    let raw = unsafe {
        let ptr = mi_stats_get_json(0, std::ptr::null_mut());
        if ptr.is_null() {
            return (0, 0);
        }
        let json = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
        libmimalloc_sys::mi_free(ptr as *mut c_void);
        json
    };
    // Fast path: find "committed": { "total": N, "peak": N, "current": N }
    if let Some(pos) = raw.find("\"committed\"") {
        let section = &raw[pos..raw.len().min(pos + 200)];
        let current = parse_json_field(section, "\"current\":");
        let peak = parse_json_field(section, "\"peak\":");
        return (current, peak);
    }
    // Fallback to full parse
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap_or_default();
    (
        v["committed"]["current"].as_i64().unwrap_or(0),
        v["committed"]["peak"].as_i64().unwrap_or(0),
    )
}

fn parse_json_field(s: &str, key: &str) -> i64 {
    if let Some(pos) = s.find(key) {
        let start = pos + key.len();
        // skip whitespace
        let trimmed = s[start..].trim_start();
        let end = trimmed
            .find(|c: char| !c.is_ascii_digit() && c != '-')
            .unwrap_or(trimmed.len());
        trimmed[..end].parse::<i64>().unwrap_or(0)
    } else {
        0
    }
}
