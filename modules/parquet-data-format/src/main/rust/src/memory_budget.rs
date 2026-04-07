/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! Plugin-level memory budget for the parquet writer plugin.
//!
//! Tracks cooperative memory usage across all active writers and sort operations.
//! Java sets the limit once at startup via `set_memory_limit`. Rust code calls
//! `try_reserve` before memory-heavy operations and `release` when done.
//!
//! This is analogous to DataFusion's MemoryPool but simpler — the parquet plugin
//! has fewer allocation points and doesn't need per-query granularity.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use vectorized_exec_spi::log_info;

/// The configured memory limit in bytes. Set once from Java.
static MEMORY_LIMIT: OnceLock<usize> = OnceLock::new();

/// Current reserved bytes across all writers and sort operations.
static RESERVED: AtomicUsize = AtomicUsize::new(0);

/// Peak reserved bytes (high-water mark).
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Set the plugin memory limit. Called once from Java at startup.
pub fn set_memory_limit(limit: usize) {
    MEMORY_LIMIT.get_or_init(|| {
        log_info!("Parquet plugin memory limit set to {}MB", limit / (1024 * 1024));
        limit
    });
}

/// Returns the configured memory limit, or usize::MAX if not set (unlimited).
pub fn memory_limit() -> usize {
    *MEMORY_LIMIT.get().unwrap_or(&usize::MAX)
}

/// Try to reserve `bytes` from the budget.
/// Returns `Ok(())` if the reservation fits within the limit,
/// or `Err(message)` if it would exceed the limit.
pub fn try_reserve(bytes: usize) -> Result<(), String> {
    loop {
        let current = RESERVED.load(Ordering::Relaxed);
        let new_total = current.saturating_add(bytes);
        if new_total > memory_limit() {
            return Err(format!(
                "Parquet plugin memory limit exceeded: requested {}B, current {}B, limit {}B",
                bytes, current, memory_limit()
            ));
        }
        // CAS: only succeed if no one else changed RESERVED since we read it
        if RESERVED.compare_exchange(current, new_total, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
            // Update peak
            PEAK.fetch_max(new_total, Ordering::Relaxed);
            return Ok(());
        }
        // CAS failed — another thread reserved concurrently, retry
    }
}

/// Release `bytes` back to the budget.
pub fn release(bytes: usize) {
    RESERVED.fetch_sub(bytes, Ordering::Relaxed);
}

/// Current reserved bytes.
pub fn reserved() -> usize {
    RESERVED.load(Ordering::Relaxed)
}

/// Peak reserved bytes since startup.
pub fn peak() -> usize {
    PEAK.load(Ordering::Relaxed)
}

/// Reset peak tracking (useful for periodic reporting).
pub fn reset_peak() {
    PEAK.store(RESERVED.load(Ordering::Relaxed), Ordering::Relaxed);
}
