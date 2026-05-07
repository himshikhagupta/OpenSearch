/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! jemalloc allocator interface: memory stats, runtime tuning, and per-plugin tracking.
//!
//! ## Per-Plugin Memory Tracking (Approach B: Tagged Allocator)
//!
//! Every allocation is prefixed with a 1-byte header storing the plugin ID.
//! On dealloc, the header is read to credit the correct plugin regardless of
//! which thread frees the memory. This handles cross-plugin buffer transfers.
//!
//! FFI convention (same as all other native bridge functions):
//!   - `>= 0` → success (the stat value in bytes, or 0 for setters)
//!   - `< 0`  → error pointer. Negate and pass to `native_error_message` / `native_error_free`.

use crate::error::{ffm_wrap, into_error_ptr};
use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::OnceLock;
use tikv_jemalloc_ctl::{epoch, epoch_mib, stats, stats::allocated_mib, stats::resident_mib};
use tikv_jemallocator::Jemalloc;

// =============================================================================
// Per-plugin tracking allocator
// =============================================================================

/// Maximum number of plugins that can be tracked. Plugin ID 0 = untagged (startup/system).
pub const MAX_PLUGINS: usize = 16;

/// Per-plugin live byte counters. Index 0 is "untagged" (allocations before any plugin registers).
static LIVE_BYTES: [AtomicUsize; MAX_PLUGINS] = {
    const ZERO: AtomicUsize = AtomicUsize::new(0);
    [ZERO; MAX_PLUGINS]
};

thread_local! {
    /// Current plugin ID for this thread. 0 = untagged (startup, JNI callback threads, etc.)
    static CURRENT_PLUGIN: Cell<u8> = const { Cell::new(0) };
}

/// The underlying jemalloc allocator we delegate to.
static JEMALLOC: Jemalloc = Jemalloc;

/// Layout for the 1-byte plugin ID header.
const HEADER_LAYOUT: Layout = Layout::new::<u8>();

/// Tracking allocator that wraps jemalloc with a 1-byte plugin ID header per allocation.
/// Dealloc reads the header to credit the original allocating plugin.
pub struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let (wrapped, offset) = match HEADER_LAYOUT.extend(layout) {
            Ok(v) => v,
            Err(_) => return std::ptr::null_mut(),
        };
        let wrapped = wrapped.pad_to_align();

        let base = JEMALLOC.alloc(wrapped);
        if base.is_null() {
            crate::log_error!("[TrackingAllocator] alloc failed: size={} align={}", layout.size(), layout.align());
            return base;
        }

        let plugin_id = CURRENT_PLUGIN.with(|c| c.get());
        *base = plugin_id;

        LIVE_BYTES[plugin_id as usize].fetch_add(layout.size(), Ordering::Relaxed);
        base.add(offset)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let (wrapped, offset) = match HEADER_LAYOUT.extend(layout) {
            Ok(v) => v,
            Err(_) => return std::ptr::null_mut(),
        };
        let wrapped = wrapped.pad_to_align();

        let base = JEMALLOC.alloc_zeroed(wrapped);
        if base.is_null() {
            crate::log_error!("[TrackingAllocator] alloc_zeroed failed: size={} align={}", layout.size(), layout.align());
            return base;
        }

        let plugin_id = CURRENT_PLUGIN.with(|c| c.get());
        *base = plugin_id;

        LIVE_BYTES[plugin_id as usize].fetch_add(layout.size(), Ordering::Relaxed);
        base.add(offset)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let (wrapped, offset) = HEADER_LAYOUT.extend(layout).unwrap_unchecked();
        let wrapped = wrapped.pad_to_align();

        let base = ptr.sub(offset);
        let plugin_id = *base;

        if (plugin_id as usize) >= MAX_PLUGINS {
            crate::log_error!(
                "[TrackingAllocator] BAD plugin_id={} in dealloc! ptr={:?} base={:?} offset={} layout.size={} layout.align={}",
                plugin_id, ptr, base, offset, layout.size(), layout.align()
            );
            // Skip counter update but still free to avoid leak
            JEMALLOC.dealloc(base, wrapped);
            return;
        }

        LIVE_BYTES[plugin_id as usize].fetch_sub(layout.size(), Ordering::Relaxed);
        JEMALLOC.dealloc(base, wrapped);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let (old_wrapped, offset) = HEADER_LAYOUT.extend(layout).unwrap_unchecked();
        let old_wrapped = old_wrapped.pad_to_align();

        let base = ptr.sub(offset);
        let plugin_id = *base;

        if (plugin_id as usize) >= MAX_PLUGINS {
            crate::log_error!(
                "[TrackingAllocator] BAD plugin_id={} in realloc! ptr={:?} base={:?} offset={} layout.size={} layout.align={} new_size={}",
                plugin_id, ptr, base, offset, layout.size(), layout.align(), new_size
            );
            // Fall back to alloc+copy+dealloc to avoid UB
            let new_ptr = self.alloc(Layout::from_size_align_unchecked(new_size, layout.align()));
            if !new_ptr.is_null() {
                std::ptr::copy_nonoverlapping(ptr, new_ptr, layout.size().min(new_size));
            }
            JEMALLOC.dealloc(base, old_wrapped);
            return new_ptr;
        }

        let new_layout = match Layout::from_size_align(new_size, layout.align()) {
            Ok(l) => l,
            Err(_) => return std::ptr::null_mut(),
        };
        let (new_wrapped, new_offset) = match HEADER_LAYOUT.extend(new_layout) {
            Ok(v) => v,
            Err(_) => return std::ptr::null_mut(),
        };
        let new_wrapped = new_wrapped.pad_to_align();

        let new_base = JEMALLOC.realloc(base, old_wrapped, new_wrapped.size());
        if new_base.is_null() {
            crate::log_error!("[TrackingAllocator] realloc failed: old_size={} new_size={} align={}", layout.size(), new_size, layout.align());
            return new_base;
        }

        // Header is preserved by realloc (same offset, same position)
        // Update counters
        LIVE_BYTES[plugin_id as usize].fetch_add(new_size, Ordering::Relaxed);
        LIVE_BYTES[plugin_id as usize].fetch_sub(layout.size(), Ordering::Relaxed);

        new_base.add(new_offset)
    }
}

// =============================================================================
// Plugin registration and binding
// =============================================================================

/// Opaque handle to a registered plugin. Wraps an internal u8 ID.
/// Obtained from `register_plugin()` — plugins store this in a `OnceLock<PluginHandle>`.
#[derive(Clone, Copy, Debug)]
pub struct PluginHandle(u8);

impl PluginHandle {
    /// Returns the internal numeric ID (for FFI reporting to Java).
    pub fn id(&self) -> u8 {
        self.0
    }
}

/// Plugin name registry. Index 0 = "untagged".
static PLUGIN_NAMES: [OnceLock<&'static str>; MAX_PLUGINS] = {
    const EMPTY: OnceLock<&'static str> = OnceLock::new();
    [EMPTY; MAX_PLUGINS]
};

/// Next plugin ID to assign. Starts at 1 (0 = untagged).
static NEXT_ID: AtomicU8 = AtomicU8::new(1);

/// Register a plugin by name. Returns a `PluginHandle` for use with `bind_thread`.
/// Panics if MAX_PLUGINS is exceeded. Call once per plugin at startup.
pub fn register_plugin(name: &'static str) -> PluginHandle {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    assert!(
        (id as usize) < MAX_PLUGINS,
        "too many plugins registered ({} >= MAX_PLUGINS {})", id, MAX_PLUGINS
    );
    PLUGIN_NAMES[id as usize].set(name).ok();
    PluginHandle(id)
}

/// Bind the current thread to a plugin. All subsequent allocations on this thread
/// will be attributed to this plugin.
pub fn bind_thread(handle: &PluginHandle) {
    CURRENT_PLUGIN.with(|c| c.set(handle.0));
}

/// Bind the current thread to a plugin by raw ID. Used by the macro expansion.
/// Prefer `bind_thread(&handle)` in application code.
pub fn bind_thread_to_plugin(plugin_id: u8) {
    debug_assert!(
        (plugin_id as usize) < MAX_PLUGINS,
        "plugin_id {} exceeds MAX_PLUGINS {}",
        plugin_id,
        MAX_PLUGINS
    );
    CURRENT_PLUGIN.with(|c| c.set(plugin_id));
}

/// Get the current thread's plugin ID.
pub fn current_plugin_id() -> u8 {
    CURRENT_PLUGIN.with(|c| c.get())
}

/// Read live bytes for a specific plugin. Lock-free atomic load.
pub fn plugin_live_bytes(handle: &PluginHandle) -> usize {
    LIVE_BYTES[handle.0 as usize].load(Ordering::Relaxed)
}

/// Read live bytes by raw ID (for FFI).
pub fn plugin_live_bytes_by_id(plugin_id: u8) -> usize {
    if (plugin_id as usize) >= MAX_PLUGINS {
        return 0;
    }
    LIVE_BYTES[plugin_id as usize].load(Ordering::Relaxed)
}

/// Get plugin name by ID (for observability/Java reporting).
pub fn plugin_name(plugin_id: u8) -> Option<&'static str> {
    PLUGIN_NAMES.get(plugin_id as usize).and_then(|l| l.get().copied())
}

/// Get number of registered plugins (excluding untagged slot 0).
pub fn registered_plugin_count() -> u8 {
    NEXT_ID.load(Ordering::Relaxed) - 1
}

// =============================================================================
// FFI: Per-plugin memory stats
// =============================================================================

/// FFI: Returns live bytes for a given plugin ID. Returns 0 if plugin_id is out of range.
#[no_mangle]
pub extern "C" fn native_plugin_live_bytes(plugin_id: u8) -> i64 {
    plugin_live_bytes_by_id(plugin_id) as i64
}

/// FFI: Bind the calling thread to a plugin ID. Returns 0 on success, negative error on failure.
#[no_mangle]
pub extern "C" fn native_bind_thread_to_plugin(plugin_id: u8) -> i64 {
    if (plugin_id as usize) >= MAX_PLUGINS {
        return into_error_ptr(format!("plugin_id {} exceeds MAX_PLUGINS {}", plugin_id, MAX_PLUGINS));
    }
    CURRENT_PLUGIN.with(|c| c.set(plugin_id));
    0
}

// =============================================================================
// Global jemalloc stats (unchanged from before)
// =============================================================================

struct StatsMib {
    epoch: epoch_mib,
    allocated: allocated_mib,
    resident: resident_mib,
}

static MIB: OnceLock<StatsMib> = OnceLock::new();

fn mib() -> &'static StatsMib {
    MIB.get_or_init(|| StatsMib {
        epoch: epoch::mib().unwrap(),
        allocated: stats::allocated::mib().unwrap(),
        resident: stats::resident::mib().unwrap(),
    })
}

/// Advances the jemalloc epoch and reads both stats atomically.
/// Logs elapsed time at debug level for performance profiling.
fn refresh_stats() -> Result<(i64, i64), String> {
    let start = std::time::Instant::now();
    let m = mib();
    m.epoch.advance().map_err(|e| format!("jemalloc epoch advance failed: {}", e))?;
    let alloc = m.allocated.read().map_err(|e| format!("jemalloc allocated read failed: {}", e))? as i64;
    let res = m.resident.read().map_err(|e| format!("jemalloc resident read failed: {}", e))? as i64;
    let elapsed_us = start.elapsed().as_micros();
    if elapsed_us > 100 {
        crate::log_info!("[jemalloc-stats] refresh_stats took {} µs (SLOW)", elapsed_us);
    } else {
        crate::log_debug!("[jemalloc-stats] refresh_stats took {} µs", elapsed_us);
    }
    Ok((alloc, res))
}

/// Returns current jemalloc allocated bytes (live malloc'd objects).
pub fn allocated_bytes() -> i64 {
    match refresh_stats() {
        Ok((alloc, _)) => alloc,
        Err(msg) => into_error_ptr(msg),
    }
}

/// Returns current jemalloc resident bytes (physical RAM used by native layer only).
pub fn resident_bytes() -> i64 {
    match refresh_stats() {
        Ok((_, res)) => res,
        Err(msg) => into_error_ptr(msg),
    }
}

/// FFI: Returns current jemalloc allocated bytes, or negative error pointer.
#[no_mangle]
pub extern "C" fn native_jemalloc_allocated_bytes() -> i64 {
    ffm_wrap("native_jemalloc_allocated_bytes", || refresh_stats().map(|(alloc, _)| alloc))
}

/// FFI: Returns current jemalloc resident bytes, or negative error pointer.
#[no_mangle]
pub extern "C" fn native_jemalloc_resident_bytes() -> i64 {
    ffm_wrap("native_jemalloc_resident_bytes", || refresh_stats().map(|(_, res)| res))
}

/// FFI: Sets dirty_decay_ms for all arenas at runtime.
#[no_mangle]
pub extern "C" fn native_jemalloc_set_dirty_decay_ms(ms: i64) -> i64 {
    ffm_wrap("native_jemalloc_set_dirty_decay_ms", || set_all_arenas(b"dirty_decay_ms\0", ms))
}

/// FFI: Sets muzzy_decay_ms for all arenas at runtime.
#[no_mangle]
pub extern "C" fn native_jemalloc_set_muzzy_decay_ms(ms: i64) -> i64 {
    ffm_wrap("native_jemalloc_set_muzzy_decay_ms", || set_all_arenas(b"muzzy_decay_ms\0", ms))
}

fn set_all_arenas(suffix: &[u8], ms: i64) -> Result<i64, String> {
    let narenas: u32 = unsafe { tikv_jemalloc_ctl::raw::read(b"arenas.narenas\0") }
        .map_err(|e| format!("failed to read arenas.narenas: {}", e))?;
    let suffix_str = std::str::from_utf8(&suffix[..suffix.len() - 1]).unwrap();
    let mut any_success = false;
    for i in 0..narenas {
        let key = format!("arena.{}.{}\0", i, suffix_str);
        if unsafe { tikv_jemalloc_ctl::raw::write(key.as_bytes(), ms as isize) }.is_ok() {
            any_success = true;
        }
    }
    if any_success {
        Ok(0)
    } else {
        Err(format!("failed to set {} on any arena", suffix_str))
    }
}

// =============================================================================
// Per-plugin metrics monitoring
// =============================================================================

use std::sync::atomic::AtomicBool;

static MONITOR_RUNNING: AtomicBool = AtomicBool::new(false);

/// Start a background thread that logs per-plugin memory metrics at the given interval.
/// No-op if already running. Thread exits when `stop_plugin_monitor()` is called.
pub fn start_plugin_monitor(interval_secs: u64) {
    if MONITOR_RUNNING.swap(true, Ordering::Relaxed) {
        return; // already running
    }
    std::thread::Builder::new()
        .name("plugin-mem-monitor".into())
        .spawn(move || {
            while MONITOR_RUNNING.load(Ordering::Relaxed) {
                // Per-plugin tagged counters
                let count = registered_plugin_count();
                let mut tagged_total: usize = 0;
                for id in 1..=count {
                    let name = plugin_name(id).unwrap_or("?");
                    let bytes = plugin_live_bytes_by_id(id);
                    tagged_total += bytes;
                    crate::log_info!(
                        "[plugin-memory] plugin={} mb={:.2}",
                        name, bytes as f64 / (1024.0 * 1024.0)
                    );
                }
                let untagged = plugin_live_bytes_by_id(0);
                tagged_total += untagged;
                crate::log_info!(
                    "[plugin-memory] plugin=untagged mb={:.2}",
                    untagged as f64 / (1024.0 * 1024.0)
                );
                crate::log_info!(
                    "[plugin-memory] plugin=tracked_total mb={:.2}",
                    tagged_total as f64 / (1024.0 * 1024.0)
                );
                // Total jemalloc stats (epoch advance + read)
                match refresh_stats() {
                    Ok((alloc, res)) => crate::log_info!(
                        "[jemalloc-total] allocated_mb={:.2} resident_mb={:.2}",
                        alloc as f64 / (1024.0 * 1024.0),
                        res as f64 / (1024.0 * 1024.0)
                    ),
                    Err(e) => crate::log_info!("[jemalloc-total] stats read failed: {}", e),
                }
                std::thread::sleep(std::time::Duration::from_secs(interval_secs));
            }
        })
        .ok();
}

/// Stop the monitoring thread.
pub fn stop_plugin_monitor() {
    MONITOR_RUNNING.store(false, Ordering::Relaxed);
}

/// FFI: Start plugin memory monitoring at the given interval (seconds).
#[no_mangle]
pub extern "C" fn native_start_plugin_monitor(interval_secs: i64) -> i64 {
    if interval_secs <= 0 {
        return into_error_ptr("interval must be positive".to_string());
    }
    start_plugin_monitor(interval_secs as u64);
    0
}

/// FFI: Stop plugin memory monitoring.
#[no_mangle]
pub extern "C" fn native_stop_plugin_monitor() -> i64 {
    stop_plugin_monitor();
    0
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[global_allocator]
    static GLOBAL: TrackingAllocator = TrackingAllocator;

    #[test]
    fn allocated_bytes_is_positive() {
        assert!(allocated_bytes() > 0);
    }

    #[test]
    fn resident_bytes_is_positive() {
        assert!(resident_bytes() > 0);
    }

    #[test]
    fn allocated_increases_after_allocation() {
        let handle = register_plugin("test_alloc_increases");
        bind_thread(&handle);
        let before = plugin_live_bytes(&handle);
        let _data: Vec<u8> = vec![42u8; 1024 * 1024];
        let after = plugin_live_bytes(&handle);
        assert!(after >= before + 1024 * 1024, "expected {after} >= {} (before + 1MB)", before + 1024 * 1024);
        bind_thread_to_plugin(0);
    }

    #[test]
    fn plugin_tracking_basic() {
        let handle = register_plugin("test_basic");
        bind_thread(&handle);
        let before = plugin_live_bytes(&handle);

        let data: Vec<u8> = vec![0u8; 1024 * 1024];
        let after = plugin_live_bytes(&handle);
        assert!(after > before, "expected {after} > {before} after 1MB alloc");

        drop(data);
        let after_drop = plugin_live_bytes(&handle);
        assert!(after_drop < after, "expected {after_drop} < {after} after drop");

        bind_thread_to_plugin(0);
    }

    #[test]
    fn cross_plugin_free_credits_original() {
        let df = register_plugin("test_cross_df");
        let pq = register_plugin("test_cross_pq");

        bind_thread(&df);
        let df_before = plugin_live_bytes(&df);

        let data: Vec<u8> = vec![42u8; 512 * 1024];
        let df_after_alloc = plugin_live_bytes(&df);
        assert!(df_after_alloc > df_before);

        // Switch to "parquet" context and free
        bind_thread(&pq);
        let pq_before = plugin_live_bytes(&pq);
        drop(data);

        let df_after_free = plugin_live_bytes(&df);
        let pq_after_free = plugin_live_bytes(&pq);

        assert!(
            df_after_free < df_after_alloc,
            "datafusion should decrease: {} < {}",
            df_after_free,
            df_after_alloc
        );
        assert_eq!(
            pq_before, pq_after_free,
            "parquet should be unchanged: {} == {}",
            pq_before,
            pq_after_free
        );

        bind_thread_to_plugin(0);
    }

    #[test]
    fn cross_plugin_free_multithreaded() {
        use std::sync::mpsc;
        use std::thread;

        let alloc_handle = register_plugin("test_mt_alloc");
        let free_handle = register_plugin("test_mt_free");
        let alloc_id = alloc_handle.id();
        let free_id = free_handle.id();

        let (tx, rx) = mpsc::channel::<Vec<u8>>();

        let alloc_thread = thread::spawn(move || {
            bind_thread_to_plugin(alloc_id);
            let before = plugin_live_bytes_by_id(alloc_id);
            let data = vec![42u8; 256 * 1024];
            let after = plugin_live_bytes_by_id(alloc_id);
            assert!(
                after >= before + 256 * 1024,
                "alloc should increase by >=256KB: before={}, after={}",
                before,
                after
            );
            tx.send(data).unwrap();
            after
        });

        let after_alloc = alloc_thread.join().unwrap();

        let free_thread = thread::spawn(move || {
            bind_thread_to_plugin(free_id);
            let before = plugin_live_bytes_by_id(free_id);
            let data = rx.recv().unwrap();
            drop(data);
            let after = plugin_live_bytes_by_id(free_id);
            assert_eq!(before, after, "free plugin should be unchanged: {} == {}", before, after);
        });

        free_thread.join().unwrap();

        let after_free = plugin_live_bytes_by_id(alloc_id);
        assert!(
            after_free < after_alloc,
            "alloc plugin should decrease after cross-thread free: {} < {}",
            after_free,
            after_alloc
        );
    }

    #[test]
    fn set_dirty_decay_ms_applies_at_runtime() {
        let rc = native_jemalloc_set_dirty_decay_ms(5000);
        assert_eq!(rc, 0, "setter should succeed, got {}", rc);
        let actual: isize =
            unsafe { tikv_jemalloc_ctl::raw::read(b"arena.0.dirty_decay_ms\0") }.unwrap();
        assert_eq!(actual, 5000);
        native_jemalloc_set_dirty_decay_ms(30000);
    }

    #[test]
    fn set_muzzy_decay_ms_applies_at_runtime() {
        let rc = native_jemalloc_set_muzzy_decay_ms(10000);
        assert_eq!(rc, 0, "setter should succeed, got {}", rc);
        let actual: isize =
            unsafe { tikv_jemalloc_ctl::raw::read(b"arena.0.muzzy_decay_ms\0") }.unwrap();
        assert_eq!(actual, 10000);
        native_jemalloc_set_muzzy_decay_ms(30000);
    }
}
