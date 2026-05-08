/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! Microbenchmark for arena-group per-plugin tracking with variable arena counts.
//! Tests allocation hot path under 16-thread contention with different arena counts.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[inline(never)]
fn alloc_and_free(size: usize) {
    let v: Vec<u8> = vec![0u8; size];
    black_box(&v);
    drop(v);
}

fn bench_variable_arenas(c: &mut Criterion) {
    use native_bridge_common::plugin_arena::{register_plugin_with_arenas, bind_thread, plugin_allocated_bytes};

    let n_threads = 16;
    let mut group = c.benchmark_group("arena_contention_16t");

    for n_arenas in [4, 8, 16, 32] {
        let pid = register_plugin_with_arenas(n_arenas).unwrap() as u8;

        group.bench_with_input(
            BenchmarkId::new("alloc_4kb", format!("{}arenas", n_arenas)),
            &n_arenas,
            |b, _| {
                b.iter_custom(|iters| {
                    let barrier = Arc::new(Barrier::new(n_threads + 1));
                    let handles: Vec<_> = (0..n_threads)
                        .map(|_| {
                            let bar = barrier.clone();
                            thread::spawn(move || {
                                bind_thread(pid).unwrap();
                                bar.wait();
                                let t0 = std::time::Instant::now();
                                for _ in 0..iters {
                                    alloc_and_free(4096);
                                }
                                t0.elapsed()
                            })
                        })
                        .collect();
                    barrier.wait();
                    let total: std::time::Duration =
                        handles.into_iter().map(|h| h.join().unwrap()).sum();
                    total / n_threads as u32
                });
            },
        );

        // Also measure stats read cost (scales with n_arenas)
        group.bench_with_input(
            BenchmarkId::new("stats_read", format!("{}arenas", n_arenas)),
            &n_arenas,
            |b, _| {
                b.iter(|| {
                    black_box(plugin_allocated_bytes(pid).unwrap());
                });
            },
        );
    }

    // Baseline: unbound (jemalloc default arena assignment)
    group.bench_function("alloc_4kb/unbound", |b| {
        bind_thread(0).unwrap();
        b.iter_custom(|iters| {
            let barrier = Arc::new(Barrier::new(n_threads + 1));
            let handles: Vec<_> = (0..n_threads)
                .map(|_| {
                    let bar = barrier.clone();
                    thread::spawn(move || {
                        bar.wait();
                        let t0 = std::time::Instant::now();
                        for _ in 0..iters {
                            alloc_and_free(4096);
                        }
                        t0.elapsed()
                    })
                })
                .collect();
            barrier.wait();
            let total: std::time::Duration =
                handles.into_iter().map(|h| h.join().unwrap()).sum();
            total / n_threads as u32
        });
    });

    // Isolate: epoch advance only
    group.bench_function("epoch_advance_only", |b| {
        b.iter(|| {
            unsafe { tikv_jemalloc_ctl::raw::update(b"epoch\0", 1u64).unwrap() };
        });
    });

    // Isolate: 32 MIB reads without epoch
    {
        let pid32 = register_plugin_with_arenas(32).unwrap() as u8;
        bind_thread(pid32).unwrap();
        let _keep: Vec<u8> = vec![0u8; 64 * 1024];
        group.bench_function("32_mib_reads_no_epoch", |b| {
            use tikv_jemalloc_ctl::raw;
            let mut small_mib = [0usize; 5];
            raw::name_to_mib(b"stats.arenas.0.small.allocated\0", &mut small_mib).unwrap();
            b.iter(|| {
                let mut total = 0usize;
                for i in 0..32 {
                    small_mib[2] = i;
                    let v: usize = unsafe { raw::read_mib(&small_mib) }.unwrap_or(0);
                    total += v;
                }
                black_box(total);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_variable_arenas);
criterion_main!(benches);
