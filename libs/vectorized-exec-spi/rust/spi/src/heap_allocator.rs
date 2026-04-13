/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! Per-plugin heap tracking via mimalloc v3 first-class heaps.
//!
//! Each plugin creates its own heap via [`create_heap`] (which auto-registers
//! it for monitoring) and wires it into thread pools via [`set_thread_heap`].
//! [`all_plugin_stats`] returns a snapshot of every registered plugin's usage.

use core::ffi::c_void;
use std::sync::{Mutex, Once};
use std::thread;
use std::time::Duration;

pub use libmimalloc_sys::mi_heap_t;
use libmimalloc_sys::{mi_heap_area_t, mi_heap_new, mi_heap_set_default, mi_heap_visit_blocks};

static REGISTRY: Mutex<Vec<(&'static str, PluginHeap)>> = Mutex::new(Vec::new());
static MONITOR_STARTED: Once = Once::new();

/// Thread-safe wrapper around a mimalloc heap pointer.
/// Safe to send/share across threads in mimalloc v3.
#[derive(Debug, Clone, Copy)]
pub struct PluginHeap {
    ptr: *mut mi_heap_t,
}

unsafe impl Send for PluginHeap {}
unsafe impl Sync for PluginHeap {}

/// Create a new mimalloc heap for a plugin and register it for monitoring.
/// Automatically starts the monitoring loop on the first call.
pub fn create_heap(name: &'static str) -> PluginHeap {
    let ptr = unsafe { mi_heap_new() };
    assert!(!ptr.is_null(), "mi_heap_new failed for '{}'", name);
    let heap = PluginHeap { ptr };
    REGISTRY.lock().unwrap().push((name, heap));
    crate::log_info!("Created plugin heap for '{}'", name);
    MONITOR_STARTED.call_once(|| {
        thread::Builder::new()
            .name("heap-monitor".into())
            .spawn(|| loop {
                thread::sleep(Duration::from_secs(1));
                for ps in all_plugin_stats() {
                    crate::log_info!(
                        "[heap-monitor] plugin='{}' used={} committed={}",
                        ps.name, ps.stats.used, ps.stats.committed
                    );
                }
            })
            .expect("Failed to spawn heap-monitor thread");
    });
    heap
}

/// Set the current thread's default mimalloc heap.
pub fn set_thread_heap(heap: PluginHeap) {
    unsafe { mi_heap_set_default(heap.ptr) };
}

/// Per-plugin memory statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeapStats {
    pub used: usize,
    pub committed: usize,
}

/// Stats snapshot for a single plugin.
#[derive(Debug, Clone)]
pub struct PluginStats {
    pub name: &'static str,
    pub stats: HeapStats,
}

/// Collect memory stats for a single heap.
pub fn heap_stats(heap: PluginHeap) -> HeapStats {
    if heap.ptr.is_null() {
        return HeapStats::default();
    }

    struct Acc { used: usize, committed: usize }

    unsafe extern "C" fn visitor(
        _heap: *const mi_heap_t,
        area: *const mi_heap_area_t,
        block: *mut c_void,
        _block_size: usize,
        arg: *mut c_void,
    ) -> bool {
        if block.is_null() {
            let acc = &mut *(arg as *mut Acc);
            let a = &*area;
            acc.used += a.used;
            acc.committed += a.committed;
        }
        true
    }

    let mut acc = Acc { used: 0, committed: 0 };
    unsafe {
        mi_heap_visit_blocks(
            heap.ptr as *const mi_heap_t,
            false,
            Some(visitor),
            &mut acc as *mut Acc as *mut c_void,
        );
    }
    HeapStats { used: acc.used, committed: acc.committed }
}

/// Return a snapshot of stats for all registered plugins.
pub fn all_plugin_stats() -> Vec<PluginStats> {
    let heaps = REGISTRY.lock().unwrap();
    heaps.iter().map(|&(name, heap)| PluginStats {
        name,
        stats: heap_stats(heap),
    }).collect()
}
