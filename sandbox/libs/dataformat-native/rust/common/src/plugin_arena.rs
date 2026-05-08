/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! Per-plugin memory tracking via jemalloc arena groups.
//!
//! Each plugin gets a dedicated set of arenas (variable count). Threads are
//! bound to a plugin's arena group via `thread.arena`, so all allocations on
//! that thread go to the plugin's arenas. jemalloc handles cross-plugin frees
//! correctly (it knows which arena owns each allocation from chunk metadata).
//!
//! Stats reads use MIB-cached mallctl keys for fast lookups (no string parsing).

use crate::error::{ffm_wrap, into_error_ptr};
use std::cell::Cell;
use std::sync::OnceLock;
use tikv_jemalloc_ctl::raw;

/// Maximum arenas per plugin.
const MAX_ARENAS_PER_PLUGIN: usize = 64;

/// Maximum number of plugins we support.
const MAX_PLUGINS: usize = 16;

/// Pre-resolved MIB templates for stats reads.
/// `stats.arenas.0.small.allocated` → 5-element MIB, index [2] is the arena slot.
/// `stats.arenas.0.large.allocated` → same structure.
struct StatsMibs {
    small_alloc: [usize; 5],
    large_alloc: [usize; 5],
    epoch: [usize; 1],
}

static STATS_MIBS: OnceLock<StatsMibs> = OnceLock::new();

fn stats_mibs() -> &'static StatsMibs {
    STATS_MIBS.get_or_init(|| {
        let mut small = [0usize; 5];
        let mut large = [0usize; 5];
        let mut epoch = [0usize; 1];
        raw::name_to_mib(b"stats.arenas.0.small.allocated\0", &mut small).unwrap();
        raw::name_to_mib(b"stats.arenas.0.large.allocated\0", &mut large).unwrap();
        raw::name_to_mib(b"epoch\0", &mut epoch).unwrap();
        StatsMibs {
            small_alloc: small,
            large_alloc: large,
            epoch,
        }
    })
}

struct PluginEntry {
    arenas: [u32; MAX_ARENAS_PER_PLUGIN],
    n_arenas: usize,
}

struct PluginArenas {
    plugins: [PluginEntry; MAX_PLUGINS],
    count: usize,
}

static REGISTRY: OnceLock<std::sync::Mutex<PluginArenas>> = OnceLock::new();

fn registry() -> &'static std::sync::Mutex<PluginArenas> {
    REGISTRY.get_or_init(|| {
        const EMPTY: PluginEntry = PluginEntry {
            arenas: [0; MAX_ARENAS_PER_PLUGIN],
            n_arenas: 0,
        };
        std::sync::Mutex::new(PluginArenas {
            plugins: [EMPTY; MAX_PLUGINS],
            count: 0,
        })
    })
}

thread_local! {
    static CURRENT_PLUGIN: Cell<u8> = const { Cell::new(0) };
}

fn create_arenas(n: usize) -> Result<Vec<u32>, String> {
    let mut indices = Vec::with_capacity(n);
    for _ in 0..n {
        let arena_idx: u32 = unsafe { raw::read(b"arenas.create\0") }
            .map_err(|e| format!("arenas.create failed: {}", e))?;
        indices.push(arena_idx);
    }
    Ok(indices)
}

/// Registers a plugin with `n_arenas` dedicated arenas.
pub fn register_plugin_with_arenas(n_arenas: usize) -> Result<i64, String> {
    if n_arenas == 0 || n_arenas > MAX_ARENAS_PER_PLUGIN {
        return Err(format!("n_arenas must be 1..{}", MAX_ARENAS_PER_PLUGIN));
    }
    let mut reg = registry().lock().map_err(|e| format!("lock poisoned: {}", e))?;
    if reg.count >= MAX_PLUGINS {
        return Err(format!("max plugins ({}) exceeded", MAX_PLUGINS));
    }
    let plugin_id = reg.count + 1;
    let arenas = create_arenas(n_arenas)?;
    for (i, &arena_idx) in arenas.iter().enumerate() {
        reg.plugins[plugin_id].arenas[i] = arena_idx;
    }
    reg.plugins[plugin_id].n_arenas = n_arenas;
    reg.count = plugin_id;
    // Eagerly init MIB cache
    let _ = stats_mibs();
    Ok(plugin_id as i64)
}

/// Registers a plugin with default 4 arenas.
pub fn register_plugin() -> Result<i64, String> {
    register_plugin_with_arenas(4)
}

/// Registers a plugin with ncpus arenas (one per core, minimizes contention).
pub fn register_plugin_auto() -> Result<i64, String> {
    let n = num_cpus::get().max(1).min(MAX_ARENAS_PER_PLUGIN);
    register_plugin_with_arenas(n)
}

/// Binds the calling thread to a plugin's arena group.
pub fn bind_thread(plugin_id: u8) -> Result<i64, String> {
    if plugin_id == 0 {
        CURRENT_PLUGIN.with(|c| c.set(0));
        unsafe { raw::write(b"thread.arena\0", 0u32) }
            .map_err(|e| format!("thread.arena reset failed: {}", e))?;
        return Ok(0);
    }

    let reg = registry().lock().map_err(|e| format!("lock poisoned: {}", e))?;
    if plugin_id as usize > reg.count {
        return Err(format!("plugin_id {} not registered (max={})", plugin_id, reg.count));
    }

    let entry = &reg.plugins[plugin_id as usize];
    let thread_hash = {
        use std::hash::{Hash, Hasher};
        let mut h = std::hash::DefaultHasher::new();
        std::thread::current().id().hash(&mut h);
        h.finish() as usize
    };
    let arena_idx = entry.arenas[thread_hash % entry.n_arenas];
    drop(reg);

    unsafe { raw::write(b"thread.arena\0", arena_idx) }
        .map_err(|e| format!("thread.arena write failed: {}", e))?;

    CURRENT_PLUGIN.with(|c| c.set(plugin_id));
    Ok(0)
}

/// Reads allocated bytes for a plugin using MIB-cached mallctl reads.
pub fn plugin_allocated_bytes(plugin_id: u8) -> Result<i64, String> {
    plugin_allocated_bytes_inner(plugin_id, true)
}

/// Reads allocated bytes without advancing epoch (caller must have advanced it).
pub fn plugin_allocated_bytes_no_epoch(plugin_id: u8) -> i64 {
    plugin_allocated_bytes_inner(plugin_id, false).unwrap_or(0)
}

fn plugin_allocated_bytes_inner(plugin_id: u8, advance_epoch: bool) -> Result<i64, String> {
    if plugin_id == 0 {
        return Err("plugin_id 0 (untagged) has no dedicated arenas".into());
    }

    let reg = registry().lock().map_err(|e| format!("lock poisoned: {}", e))?;
    if plugin_id as usize > reg.count {
        return Err(format!("plugin_id {} not registered", plugin_id));
    }
    let entry = &reg.plugins[plugin_id as usize];
    let n = entry.n_arenas;
    let mut arena_indices = [0u32; MAX_ARENAS_PER_PLUGIN];
    arena_indices[..n].copy_from_slice(&entry.arenas[..n]);
    drop(reg);

    let mibs = stats_mibs();

    if advance_epoch {
        unsafe { raw::update_mib(&mibs.epoch, 1u64) }
            .map_err(|e| format!("epoch advance failed: {}", e))?;
    }

    let mut total: i64 = 0;
    for i in 0..n {
        let idx = arena_indices[i] as usize;

        let mut small_mib = mibs.small_alloc;
        small_mib[2] = idx;
        let small: usize = unsafe { raw::read_mib(&small_mib) }.unwrap_or(0);

        let mut large_mib = mibs.large_alloc;
        large_mib[2] = idx;
        let large: usize = unsafe { raw::read_mib(&large_mib) }.unwrap_or(0);

        total += (small + large) as i64;
    }
    Ok(total)
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFI exports
// ═══════════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn native_plugin_register_with_arenas(n_arenas: i64) -> i64 {
    ffm_wrap("native_plugin_register_with_arenas", || {
        register_plugin_with_arenas(n_arenas as usize)
    })
}

#[no_mangle]
pub extern "C" fn native_plugin_register() -> i64 {
    ffm_wrap("native_plugin_register", register_plugin)
}

#[no_mangle]
pub extern "C" fn native_plugin_bind_thread(plugin_id: i64) -> i64 {
    ffm_wrap("native_plugin_bind_thread", || bind_thread(plugin_id as u8))
}

#[no_mangle]
pub extern "C" fn native_plugin_allocated_bytes(plugin_id: i64) -> i64 {
    ffm_wrap("native_plugin_allocated_bytes", || {
        plugin_allocated_bytes(plugin_id as u8)
    })
}

#[no_mangle]
pub extern "C" fn native_plugin_count() -> i64 {
    match registry().lock() {
        Ok(reg) => reg.count as i64,
        Err(_) => into_error_ptr("lock poisoned".into()),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Monitoring
// ═══════════════════════════════════════════════════════════════════════════════

static MONITOR_RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Start periodic per-plugin + jemalloc total metrics logging.
/// Safe to call multiple times — only the first call spawns the thread.
#[no_mangle]
pub extern "C" fn native_start_plugin_monitor(interval_secs: i64) {
    use std::sync::atomic::Ordering;
    if MONITOR_RUNNING.swap(true, Ordering::SeqCst) {
        return;
    }
    let interval = std::time::Duration::from_secs(interval_secs.max(1) as u64);
    std::thread::Builder::new()
        .name("plugin-mem-monitor".into())
        .spawn(move || {
            loop {
                std::thread::sleep(interval);
                if unsafe { raw::update(b"epoch\0", 1u64) }.is_err() {
                    continue;
                }

                let count = native_plugin_count();
                let mut tracked_total: i64 = 0;
                for pid in 1..=count {
                    let bytes = plugin_allocated_bytes_no_epoch(pid as u8);
                    tracked_total += bytes;
                    crate::log_info!(
                        "[plugin-memory] plugin={} mb={:.2}",
                        pid, bytes as f64 / (1024.0 * 1024.0)
                    );
                }
                crate::log_info!(
                    "[plugin-memory] plugin=tracked_total mb={:.2}",
                    tracked_total as f64 / (1024.0 * 1024.0)
                );

                let allocated: usize = unsafe { raw::read(b"stats.allocated\0") }.unwrap_or(0);
                let resident: usize = unsafe { raw::read(b"stats.resident\0") }.unwrap_or(0);
                crate::log_info!(
                    "[jemalloc-total] allocated_mb={:.2} resident_mb={:.2}",
                    allocated as f64 / (1024.0 * 1024.0),
                    resident as f64 / (1024.0 * 1024.0)
                );
            }
        })
        .expect("Failed to spawn plugin-mem-monitor thread");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[global_allocator]
    static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

    #[test]
    fn register_and_bind() {
        let pid = register_plugin().unwrap();
        assert!(pid > 0);
        bind_thread(pid as u8).unwrap();
        let _data: Vec<u8> = vec![42u8; 64 * 1024];
        let bytes = plugin_allocated_bytes(pid as u8).unwrap();
        assert!(bytes >= 64 * 1024, "expected >= 64KB, got {}", bytes);
        bind_thread(0).unwrap();
    }

    #[test]
    fn variable_arenas() {
        let pid = register_plugin_with_arenas(16).unwrap() as u8;
        bind_thread(pid).unwrap();
        let _data: Vec<u8> = vec![0u8; 128 * 1024];
        let bytes = plugin_allocated_bytes(pid).unwrap();
        assert!(bytes >= 128 * 1024, "expected >= 128KB, got {}", bytes);
        bind_thread(0).unwrap();
    }

    #[test]
    fn cross_plugin_free_works() {
        let p1 = register_plugin_with_arenas(8).unwrap() as u8;
        let p2 = register_plugin_with_arenas(4).unwrap() as u8;
        bind_thread(p1).unwrap();
        let data: Vec<u8> = vec![99u8; 512 * 1024];
        bind_thread(p2).unwrap();
        drop(data);
        let p1_after = plugin_allocated_bytes(p1).unwrap();
        assert!(p1_after >= 0);
    }
}
