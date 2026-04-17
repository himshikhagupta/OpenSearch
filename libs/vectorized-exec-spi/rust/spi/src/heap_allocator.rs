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

use core::ffi::{c_char, c_void};
use std::sync::{Mutex, Once};
use std::thread;
use std::time::Duration;

pub use libmimalloc_sys::mi_heap_t;
use libmimalloc_sys::{mi_heap_area_t, mi_heap_new, mi_heap_set_default, mi_heap_visit_blocks};

// v3 global stats JSON API — not yet in libmimalloc-sys bindings
extern "C" {
    fn mi_stats_get_json(buf_size: usize, buf: *mut c_char) -> *mut c_char;
    fn mi_stats_merge();
    fn mi_free(p: *mut c_void);
}

/// Returns global mimalloc stats as a flat JSON string.
/// Parses the nested mi_stats_get_json output and extracts key fields.
/// Format: {"current_commit":N,"peak_commit":N}
pub fn global_mimalloc_stats_json() -> String {
    unsafe { mi_stats_merge() };
    let raw = unsafe {
        let ptr = mi_stats_get_json(0, std::ptr::null_mut());
        if ptr.is_null() {
            return r#"{"current_commit":0,"peak_commit":0}"#.to_string();
        }
        let json = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
        mi_free(ptr as *mut c_void);
        json
    };
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap_or_default();
    let current_commit = v["committed"]["current"].as_u64().unwrap_or(0);
    let peak_commit = v["committed"]["peak"].as_u64().unwrap_or(0);
    format!("{{\"current_commit\":{},\"peak_commit\":{}}}", current_commit, peak_commit)
}

static REGISTRY: Mutex<Vec<(&'static str, PluginHeap)>> = Mutex::new(Vec::new());
static MONITOR_STARTED: Once = Once::new();
static MONITOR_RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

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
            .spawn(|| {
                while MONITOR_RUNNING.load(std::sync::atomic::Ordering::Relaxed) {
                    thread::sleep(Duration::from_secs(1));
                    for ps in all_plugin_stats() {
                        crate::log_info!(
                            "[heap-monitor] plugin='{}' used={}KB committed={}KB",
                            ps.name, ps.stats.used / 1024, ps.stats.committed / 1024
                        );
                    }
                }
            })
            .expect("Failed to spawn heap-monitor thread");
    });
    heap
}

/// Set the current thread's default mimalloc heap permanently.
/// Use only on threads you own (e.g. Tokio worker threads via on_thread_start).
pub fn set_thread_heap(heap: PluginHeap) {
    unsafe { mi_heap_set_default(heap.ptr) };
}

/// Stop the background heap-monitor thread.
pub fn stop_monitor() {
    MONITOR_RUNNING.store(false, std::sync::atomic::Ordering::Relaxed);
}

/// Temporarily set the current thread's default mimalloc heap.
/// Returns a guard that restores the previous heap when dropped.
/// Use on borrowed threads (e.g. OpenSearch thread pool threads called via JNI).
pub fn scoped_thread_heap(heap: PluginHeap) -> HeapGuard {
    let prev = unsafe { mi_heap_set_default(heap.ptr) };
    HeapGuard { prev }
}

/// RAII guard that restores the previous default mimalloc heap on drop.
pub struct HeapGuard {
    prev: *mut mi_heap_t,
}

unsafe impl Send for HeapGuard {}

impl Drop for HeapGuard {
    fn drop(&mut self) {
        unsafe { mi_heap_set_default(self.prev) };
    }
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

/// Collect memory stats for a single heap via mi_heap_visit_blocks.
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
            acc.used += a.used * a.block_size;
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

#[cfg(test)]
mod tests {
    use super::*;
    use libmimalloc_sys::{mi_heap_malloc, mi_free};
    use std::thread;

    /// Allocate `size` bytes directly on the given mimalloc heap.
    /// Returns the raw pointer. Caller must free with `mi_free`.
    unsafe fn heap_alloc(heap: PluginHeap, size: usize) -> *mut c_void {
        mi_heap_malloc(heap.ptr, size)
    }

    #[test]
    fn test_single_plugin_tracks_allocations() {
        let heap = create_heap("test-plugin-single");

        let ptr = unsafe { heap_alloc(heap, 64 * 1024) };
        assert!(!ptr.is_null());

        let stats = heap_stats(heap);
        assert!(stats.used >= 64 * 1024, "used={} should be >= 64KB", stats.used);
        assert!(stats.committed >= stats.used, "committed={} should be >= used={}", stats.committed, stats.used);

        unsafe { mi_free(ptr) };
    }

    #[test]
    fn test_multiple_plugins_track_independently() {
        let heap_a = create_heap("test-plugin-A");
        let heap_b = create_heap("test-plugin-B");

        let ptr_a = unsafe { heap_alloc(heap_a, 128 * 1024) };
        let ptr_b = unsafe { heap_alloc(heap_b, 256 * 1024) };

        let stats_a = heap_stats(heap_a);
        let stats_b = heap_stats(heap_b);

        assert!(stats_a.used >= 128 * 1024, "plugin A used={} should be >= 128KB", stats_a.used);
        assert!(stats_b.used >= 256 * 1024, "plugin B used={} should be >= 256KB", stats_b.used);
        assert!(stats_b.used > stats_a.used, "plugin B used={} should be > plugin A used={}", stats_b.used, stats_a.used);

        unsafe { mi_free(ptr_a); mi_free(ptr_b); }
    }

    #[test]
    fn test_stats_decrease_after_deallocation() {
        let heap = create_heap("test-plugin-dealloc");

        let ptr = unsafe { heap_alloc(heap, 512 * 1024) };
        let used_while_alive = heap_stats(heap).used;
        assert!(used_while_alive >= 512 * 1024, "used={} should be >= 512KB while allocated", used_while_alive);

        unsafe { mi_free(ptr) };
        let used_after_free = heap_stats(heap).used;
        assert!(used_after_free < used_while_alive, "used after free={} should be < used before free={}", used_after_free, used_while_alive);
    }

    #[test]
    fn test_all_plugin_stats_contains_registered_plugins() {
        let _heap = create_heap("test-plugin-registry-check");

        let all = all_plugin_stats();
        let found = all.iter().any(|ps| ps.name == "test-plugin-registry-check");
        assert!(found, "all_plugin_stats should contain 'test-plugin-registry-check'");
    }

    #[test]
    fn test_null_heap_returns_zero_stats() {
        let null_heap = PluginHeap { ptr: std::ptr::null_mut() };
        let stats = heap_stats(null_heap);
        assert_eq!(stats.used, 0);
        assert_eq!(stats.committed, 0);
    }

    #[test]
    fn test_cross_thread_heap_allocation() {
        let heap = create_heap("test-plugin-cross-thread");

        // Allocate on a different thread using the same heap
        struct SendPtr(*mut c_void);
        unsafe impl Send for SendPtr {}

        let wrapped = thread::Builder::new()
            .name("cross-thread-worker".into())
            .spawn(move || SendPtr(unsafe { heap_alloc(heap, 128 * 1024) }))
            .unwrap()
            .join()
            .unwrap();

        let stats = heap_stats(heap);
        assert!(stats.used >= 128 * 1024, "used={} should be >= 128KB after cross-thread alloc", stats.used);

        unsafe { mi_free(wrapped.0) };
    }

    #[test]
    fn test_cross_plugin_alloc_and_free() {
        let heap_a = create_heap("test-cross-plugin-A");
        let heap_b = create_heap("test-cross-plugin-B");

        // Phase 1: Plugin A allocates
        let ptr = unsafe { heap_alloc(heap_a, 256 * 1024) };
        assert!(!ptr.is_null());

        let a_after_alloc = heap_stats(heap_a);
        let b_after_alloc = heap_stats(heap_b);
        println!("After A allocs:  A used={} committed={}  B used={} committed={}",
            a_after_alloc.used, a_after_alloc.committed, b_after_alloc.used, b_after_alloc.committed);
        assert!(a_after_alloc.used >= 256 * 1024,
            "A used={} should be >= 256KB after alloc", a_after_alloc.used);
        assert_eq!(b_after_alloc.used, 0, "B should have 0 used");

        // Phase 2: Plugin B uses the ptr (reads it) on its own thread
        let ptr_addr = ptr as usize;
        let b = heap_b;
        thread::Builder::new()
            .name("plugin-b-reader".into())
            .spawn(move || {
                set_thread_heap(b);
                // Simulate reading the data (touch the memory)
                let slice = unsafe { std::slice::from_raw_parts(ptr_addr as *const u8, 256 * 1024) };
                let _ = slice.iter().sum::<u8>();
            })
            .unwrap()
            .join()
            .unwrap();

        let a_after_use = heap_stats(heap_a);
        let b_after_use = heap_stats(heap_b);
        println!("After B reads:   A used={} committed={}  B used={} committed={}",
            a_after_use.used, a_after_use.committed, b_after_use.used, b_after_use.committed);
        // A still owns the memory, B just read it
        assert!(a_after_use.used >= 256 * 1024, "A should still own the memory");
        assert_eq!(b_after_use.used, 0, "B should still have 0 used (only read)");

        // Phase 3: Plugin B frees the ptr on its own thread
        let ptr_addr2 = ptr as usize;
        let b2 = heap_b;
        thread::Builder::new()
            .name("plugin-b-freer".into())
            .spawn(move || {
                set_thread_heap(b2);
                unsafe { mi_free(ptr_addr2 as *mut c_void) };
            })
            .unwrap()
            .join()
            .unwrap();

        let a_after_free = heap_stats(heap_a);
        let b_after_free = heap_stats(heap_b);
        println!("After B frees:   A used={} committed={}  B used={} committed={}",
            a_after_free.used, a_after_free.committed, b_after_free.used, b_after_free.committed);

        // B never allocated — freeing A's pointer doesn't charge B
        assert_eq!(b_after_free.used, 0,
            "B used={} should be 0 (freed A's memory, not its own)", b_after_free.used);
    }

    /// Simulates the real plugin thread model:
    /// - Tokio IO + CPU workers with set_thread_heap (permanent)
    /// - Rayon pool workers with set_thread_heap (permanent)
    /// - Each worker allocates via mimalloc heap
    /// Verifies all allocations across all threads are tracked in the plugin's heap stats.
    ///
    /// Note: In the real .so, #[global_allocator] = MiMalloc routes Vec/Box through
    /// mi_malloc → thread default heap. In tests, we use mi_heap_malloc directly
    /// since the test binary uses the system allocator.
    #[tokio::test]
    async fn test_tokio_and_rayon_threads_tracked() {
        let heap = create_heap("test-tokio-rayon");
        let alloc_size: usize = 64 * 1024;
        let num_tokio_tasks = 4;
        let num_rayon_tasks = 4;

        // Baseline: capture heap usage before any allocations
        let baseline = heap_stats(heap);
        println!("Baseline: used={}KB committed={}KB",
            baseline.used / 1024, baseline.committed / 1024);

        // Phase 1: Tokio tasks — simulate IO/CPU workers
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .on_thread_start(move || { set_thread_heap(heap); })
            .enable_all()
            .build()
            .unwrap();

        let heap_ptr_val = heap.ptr as usize;
        let tokio_ptrs: Vec<usize> = rt.spawn(async move {
            let mut handles = Vec::new();
            for _ in 0..num_tokio_tasks {
                let hp = heap_ptr_val;
                handles.push(tokio::spawn(async move {
                    let ptr = unsafe { mi_heap_malloc(hp as *mut mi_heap_t, alloc_size) };
                    assert!(!ptr.is_null());
                    ptr as usize
                }));
            }
            let mut ptrs = Vec::new();
            for h in handles { ptrs.push(h.await.unwrap()); }
            ptrs
        }).await.unwrap();

        let after_tokio = heap_stats(heap);
        let tokio_delta = after_tokio.used - baseline.used;
        println!("After tokio: used={}KB (delta={}KB)",
            after_tokio.used / 1024, tokio_delta / 1024);
        let tokio_expected = num_tokio_tasks * alloc_size;
        let margin = tokio_expected / 10; // 10% buffer
        assert!(tokio_delta >= tokio_expected && tokio_delta <= tokio_expected + margin,
            "tokio delta={} should be between {} and {} from {} tokio tasks",
            tokio_delta, tokio_expected, tokio_expected + margin, num_tokio_tasks);

        // Phase 2: Rayon pool — simulate merge workers
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .start_handler(move |_| { set_thread_heap(heap); })
            .build()
            .unwrap();

        // warm up threads so internal allocations settle, this memory goes to default pool since heap is not set at this point
        pool.install(|| {}); 
        let before_rayon = heap_stats(heap);
        println!("Before rayon thread start: used={}KB committed={}KB",
                 before_rayon.used / 1024, before_rayon.committed / 1024);

        let rayon_ptrs: Vec<usize> = pool.install(|| {
            use rayon::prelude::*;
            (0..num_rayon_tasks).into_par_iter().map(|_| {
                let ptr = unsafe { mi_heap_malloc(heap_ptr_val as *mut mi_heap_t, alloc_size) };
                assert!(!ptr.is_null());
                ptr as usize
            }).collect()
        });

        let after_rayon = heap_stats(heap);
        let rayon_delta = after_rayon.used - before_rayon.used;
        let rayon_expected = num_rayon_tasks * alloc_size;
        let rayon_margin = rayon_expected / 10;
        println!("After rayon: used={}KB (rayon_delta={}KB)",
            after_rayon.used / 1024, rayon_delta / 1024);
        assert!(rayon_delta >= rayon_expected && rayon_delta <= rayon_expected + rayon_margin,
            "rayon delta={} should be between {} and {} from {} rayon tasks",
            rayon_delta, rayon_expected, rayon_expected + rayon_margin, num_rayon_tasks);

        // Cleanup: free on a thread with the heap set
        let all_ptrs: Vec<usize> = tokio_ptrs.into_iter().chain(rayon_ptrs).collect();
        std::thread::Builder::new()
            .name("cleanup".into())
            .spawn(move || {
                set_thread_heap(heap);
                for addr in all_ptrs {
                    unsafe { mi_free(addr as *mut c_void) };
                }
            })
            .unwrap()
            .join()
            .unwrap();

        rt.shutdown_background();
        stop_monitor();
        // Wait for monitor thread to wake and exit (sleeps 1s in loop)
        std::thread::sleep(Duration::from_millis(1100));
    }
}
