// Microbenchmark: TrackingAllocator overhead vs raw jemalloc
// Run: cargo run --release -p bench-allocator
//
// Tests alloc/free in tight loops across multiple scenarios to isolate
// which component of the TrackingAllocator causes the most overhead.

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tikv_jemallocator::Jemalloc;

static JEMALLOC: Jemalloc = Jemalloc;

// ============================================================================
// Scenario 1: Raw jemalloc (baseline)
// ============================================================================

unsafe fn bench_raw(layout: Layout, iters: usize) {
    for _ in 0..iters {
        let ptr = JEMALLOC.alloc(layout);
        black_box(ptr);
        JEMALLOC.dealloc(ptr, layout);
    }
}

// ============================================================================
// Scenario 2: TrackingAllocator (current implementation — header + atomic)
// ============================================================================

const MAX_PLUGINS: usize = 16;
static LIVE_BYTES: [AtomicUsize; MAX_PLUGINS] = {
    const ZERO: AtomicUsize = AtomicUsize::new(0);
    [ZERO; MAX_PLUGINS]
};
const HEADER_LAYOUT: Layout = Layout::new::<u8>();

thread_local! {
    static CURRENT_PLUGIN: Cell<u8> = const { Cell::new(1) };
}

unsafe fn bench_tracking(layout: Layout, iters: usize) {
    for _ in 0..iters {
        // alloc
        let (wrapped, offset) = HEADER_LAYOUT.extend(layout).unwrap_unchecked();
        let wrapped = wrapped.pad_to_align();
        let base = JEMALLOC.alloc(wrapped);
        let plugin_id = CURRENT_PLUGIN.with(|c| c.get());
        *base = plugin_id;
        LIVE_BYTES[plugin_id as usize].fetch_add(layout.size(), Ordering::Relaxed);
        let ptr = base.add(offset);
        black_box(ptr);

        // dealloc
        let (wrapped2, offset2) = HEADER_LAYOUT.extend(layout).unwrap_unchecked();
        let wrapped2 = wrapped2.pad_to_align();
        let base2 = ptr.sub(offset2);
        let pid = *base2;
        LIVE_BYTES[pid as usize].fetch_sub(layout.size(), Ordering::Relaxed);
        JEMALLOC.dealloc(base2, wrapped2);
    }
}

// ============================================================================
// Scenario 3: Trailer approach (no header padding, plugin_id after user data)
// ============================================================================

unsafe fn bench_trailer(layout: Layout, iters: usize) {
    let total = layout.size() + 1;
    let wrapped = Layout::from_size_align_unchecked(total, layout.align());
    for _ in 0..iters {
        let ptr = JEMALLOC.alloc(wrapped);
        let plugin_id = CURRENT_PLUGIN.with(|c| c.get());
        *ptr.add(layout.size()) = plugin_id;
        LIVE_BYTES[plugin_id as usize].fetch_add(layout.size(), Ordering::Relaxed);
        black_box(ptr);

        // dealloc
        let pid = *ptr.add(layout.size());
        LIVE_BYTES[pid as usize].fetch_sub(layout.size(), Ordering::Relaxed);
        JEMALLOC.dealloc(ptr, wrapped);
    }
}

// ============================================================================
// Scenario 4: Only TLS read (isolate TLS cost)
// ============================================================================

unsafe fn bench_tls_only(layout: Layout, iters: usize) {
    for _ in 0..iters {
        let ptr = JEMALLOC.alloc(layout);
        let _pid = CURRENT_PLUGIN.with(|c| c.get());
        black_box(ptr);
        let _pid2 = CURRENT_PLUGIN.with(|c| c.get());
        JEMALLOC.dealloc(ptr, layout);
    }
}

// ============================================================================
// Scenario 5: Only atomic (isolate atomic contention cost)
// ============================================================================

unsafe fn bench_atomic_only(layout: Layout, iters: usize) {
    for _ in 0..iters {
        let ptr = JEMALLOC.alloc(layout);
        LIVE_BYTES[1].fetch_add(layout.size(), Ordering::Relaxed);
        black_box(ptr);
        LIVE_BYTES[1].fetch_sub(layout.size(), Ordering::Relaxed);
        JEMALLOC.dealloc(ptr, layout);
    }
}

// ============================================================================
// Scenario 6: Layout::extend overhead only
// ============================================================================

unsafe fn bench_layout_only(layout: Layout, iters: usize) {
    for _ in 0..iters {
        let (wrapped, offset) = HEADER_LAYOUT.extend(layout).unwrap_unchecked();
        let wrapped = wrapped.pad_to_align();
        let base = JEMALLOC.alloc(wrapped);
        let ptr = base.add(offset);
        black_box(ptr);
        let (wrapped2, _) = HEADER_LAYOUT.extend(layout).unwrap_unchecked();
        let wrapped2 = wrapped2.pad_to_align();
        JEMALLOC.dealloc(base, wrapped2);
    }
}

// ============================================================================
// Scenario 7: Sharded atomics (per-thread slot, no contention)
// ============================================================================

static SHARDED: [[AtomicUsize; 64]; MAX_PLUGINS] = {
    const ZERO: AtomicUsize = AtomicUsize::new(0);
    const ROW: [AtomicUsize; 64] = [ZERO; 64];
    [ROW; MAX_PLUGINS]
};

thread_local! {
    static THREAD_SLOT: Cell<usize> = const { Cell::new(0) };
}

static SLOT_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn assign_slot() -> usize {
    SLOT_COUNTER.fetch_add(1, Ordering::Relaxed) % 64
}

unsafe fn bench_trailer_sharded(layout: Layout, iters: usize) {
    let slot = THREAD_SLOT.with(|s| {
        if s.get() == 0 {
            let v = assign_slot().max(1);
            s.set(v);
            v
        } else {
            s.get()
        }
    });
    let total = layout.size() + 1;
    let wrapped = Layout::from_size_align_unchecked(total, layout.align());
    for _ in 0..iters {
        let ptr = JEMALLOC.alloc(wrapped);
        let plugin_id = CURRENT_PLUGIN.with(|c| c.get());
        *ptr.add(layout.size()) = plugin_id;
        SHARDED[plugin_id as usize][slot].fetch_add(layout.size(), Ordering::Relaxed);
        black_box(ptr);

        let pid = *ptr.add(layout.size());
        SHARDED[pid as usize][slot].fetch_sub(layout.size(), Ordering::Relaxed);
        JEMALLOC.dealloc(ptr, wrapped);
    }
}

// ============================================================================
// Scenario 8: Thread-local counters — Option A (registry, exact on read)
// ============================================================================

use std::sync::atomic::AtomicI64;
use std::sync::Mutex;

// Registry: each thread registers a pointer to its local counters
struct ThreadCounters {
    deltas: [AtomicI64; MAX_PLUGINS],
}

static REGISTRY: Mutex<Vec<&'static ThreadCounters>> = Mutex::new(Vec::new());

thread_local! {
    static MY_COUNTERS: &'static ThreadCounters = {
        let counters: &'static ThreadCounters = Box::leak(Box::new(ThreadCounters {
            deltas: std::array::from_fn(|_| AtomicI64::new(0)),
        }));
        REGISTRY.lock().unwrap().push(counters);
        counters
    };
}

unsafe fn bench_threadlocal_registry(layout: Layout, iters: usize) {
    let total = layout.size() + 1;
    let wrapped = Layout::from_size_align_unchecked(total, layout.align());
    for _ in 0..iters {
        let ptr = JEMALLOC.alloc(wrapped);
        let plugin_id = CURRENT_PLUGIN.with(|c| c.get());
        *ptr.add(layout.size()) = plugin_id;
        MY_COUNTERS.with(|c| c.deltas[plugin_id as usize].fetch_add(layout.size() as i64, Ordering::Relaxed));
        black_box(ptr);

        let pid = *ptr.add(layout.size());
        MY_COUNTERS.with(|c| c.deltas[pid as usize].fetch_sub(layout.size() as i64, Ordering::Relaxed));
        JEMALLOC.dealloc(ptr, wrapped);
    }
}

// Read function for Option A (not benchmarked in hot path, but shown for completeness)
#[allow(dead_code)]
fn read_live_bytes_registry(plugin_id: usize) -> i64 {
    REGISTRY.lock().unwrap().iter().map(|c| c.deltas[plugin_id].load(Ordering::Relaxed)).sum()
}

// ============================================================================
// Scenario 9: Thread-local counters — Option B (flush every N ops, no registry)
// ============================================================================

const FLUSH_INTERVAL: usize = 256;

static GLOBAL_BYTES: [AtomicI64; MAX_PLUGINS] = {
    const ZERO: AtomicI64 = AtomicI64::new(0);
    [ZERO; MAX_PLUGINS]
};

thread_local! {
    static LOCAL_DELTAS: [Cell<i64>; MAX_PLUGINS] = const { [const { Cell::new(0) }; MAX_PLUGINS] };
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[inline(always)]
unsafe fn flush_if_needed() {
    ALLOC_COUNT.with(|c| {
        let count = c.get() + 1;
        if count >= FLUSH_INTERVAL {
            LOCAL_DELTAS.with(|d| {
                for i in 0..MAX_PLUGINS {
                    let val = d[i].get();
                    if val != 0 {
                        GLOBAL_BYTES[i].fetch_add(val, Ordering::Relaxed);
                        d[i].set(0);
                    }
                }
            });
            c.set(0);
        } else {
            c.set(count);
        }
    });
}

unsafe fn bench_threadlocal_flush(layout: Layout, iters: usize) {
    let total = layout.size() + 1;
    let wrapped = Layout::from_size_align_unchecked(total, layout.align());
    for _ in 0..iters {
        let ptr = JEMALLOC.alloc(wrapped);
        let plugin_id = CURRENT_PLUGIN.with(|c| c.get());
        *ptr.add(layout.size()) = plugin_id;
        LOCAL_DELTAS.with(|d| d[plugin_id as usize].set(d[plugin_id as usize].get() + layout.size() as i64));
        flush_if_needed();
        black_box(ptr);

        let pid = *ptr.add(layout.size());
        LOCAL_DELTAS.with(|d| d[pid as usize].set(d[pid as usize].get() - layout.size() as i64));
        flush_if_needed();
        JEMALLOC.dealloc(ptr, wrapped);
    }
}

// ============================================================================
// Runner
// ============================================================================

fn run_single_threaded(name: &str, f: unsafe fn(Layout, usize), layout: Layout, iters: usize) {
    // warmup
    unsafe { f(layout, iters / 10) };

    let start = Instant::now();
    unsafe { f(layout, iters) };
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() as f64 / iters as f64;
    println!("  {:<30} {:>8.1} ns/op  ({:.2}ms total)", name, ns_per_op, elapsed.as_secs_f64() * 1000.0);
}

fn run_multi_threaded(name: &str, f: unsafe fn(Layout, usize), layout: Layout, iters: usize, threads: usize) {
    let per_thread = iters / threads;

    // warmup
    unsafe { f(layout, per_thread / 10) };

    let start = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            std::thread::spawn(move || {
                unsafe { f(layout, per_thread) };
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() as f64 / iters as f64;
    println!("  {:<30} {:>8.1} ns/op  ({:.2}ms total, {} threads)", name, ns_per_op, elapsed.as_secs_f64() * 1000.0, threads);
}

fn main() {
    let iters = 2_000_000;
    let threads = 16;

    let sizes: &[(usize, usize, &str)] = &[
        (64, 8, "64B/align8 (HashMap entry)"),
        (1024, 8, "1KB/align8 (small buffer)"),
        (4096, 64, "4KB/align64 (Arrow buffer)"),
        (65536, 64, "64KB/align64 (large Arrow)"),
    ];

    for &(size, align, desc) in sizes {
        let layout = Layout::from_size_align(size, align).unwrap();
        println!("\n{}", "=".repeat(70));
        println!("  Size: {} | Align: {} — {}", size, align, desc);
        println!("{}", "=".repeat(70));

        println!("\n  --- Single-threaded (1 thread, {} iters) ---", iters);
        run_single_threaded("raw jemalloc", bench_raw, layout, iters);
        run_single_threaded("tracking (current)", bench_tracking, layout, iters);
        run_single_threaded("trailer + shared atomic", bench_trailer, layout, iters);
        run_single_threaded("trailer + sharded atomic", bench_trailer_sharded, layout, iters);
        run_single_threaded("TLS only", bench_tls_only, layout, iters);
        run_single_threaded("atomic only", bench_atomic_only, layout, iters);
        run_single_threaded("Layout::extend only", bench_layout_only, layout, iters);
        run_single_threaded("Option A (registry)", bench_threadlocal_registry, layout, iters);
        run_single_threaded("Option B (flush/256)", bench_threadlocal_flush, layout, iters);

        println!("\n  --- Multi-threaded ({} threads, {} total iters) ---", threads, iters);
        run_multi_threaded("raw jemalloc", bench_raw, layout, iters, threads);
        run_multi_threaded("tracking (current)", bench_tracking, layout, iters, threads);
        run_multi_threaded("trailer + shared atomic", bench_trailer, layout, iters, threads);
        run_multi_threaded("trailer + sharded atomic", bench_trailer_sharded, layout, iters, threads);
        run_multi_threaded("atomic only", bench_atomic_only, layout, iters, threads);
        run_multi_threaded("Option A (registry)", bench_threadlocal_registry, layout, iters, threads);
        run_multi_threaded("Option B (flush/256)", bench_threadlocal_flush, layout, iters, threads);
    }
}
