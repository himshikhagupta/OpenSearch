/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// WHY THIS FILE EXISTS
// ═══════════════════════════════════════════════════════════════════════════════
//
// OpenSearch loads plugins and modules in isolated Java classloaders. Each plugin
// gets its own URLClassLoader (child), while shared libraries like vectorized-exec-spi
// live in the parent classloader.
//
// We want ONE native .so/.dylib shared by all plugins (engine-datafusion, parquet-data-format,
// etc.) so they share a single mimalloc allocator — memory allocated in one module can be
// safely freed in another without segfaults.
//
// THE PROBLEM:
//   JVM rule: native libraries are associated with the classloader that loaded them.
//   Native method resolution ONLY searches libraries loaded by that SAME classloader.
//   It does NOT walk up to parent classloaders.
//
//   So if we load the .so from the parent classloader (spi), but NativeBridge.java
//   lives in a child classloader (plugin), JVM can't find the native functions:
//
//     NativeBridge (child CL A) → "where is Java_..._getVersionInfo?" → searches child CL A's
//     libraries → nothing loaded there → UnsatisfiedLinkError!
//
//   And we can't load the same .so from multiple child classloaders either:
//     "Native Library already loaded in another classloader"
//
// THE SOLUTION:
//   1. Load the .so ONCE from the parent classloader (spi's NativeLibraryLoader)
//   2. The .so has ONE native method reachable by JVM: registerClassNative()
//      (because NativeLibraryLoader is in the same parent CL that loaded the .so)
//   3. Each plugin calls: NativeLibraryLoader.registerClass(NativeBridge.class)
//   4. Rust receives the Class<?> object, reflects over its native methods,
//      looks up the matching Java_*_* symbols via dlsym, and calls JNI's
//      RegisterNatives to wire them up
//   5. After RegisterNatives, JVM has direct function pointers for each native
//      method — no more classloader-based lookup needed
//
// RESULT:
//   - Single .so, single mimalloc, safe cross-module alloc/free
//   - Native method declarations stay in each plugin's Java code
//   - No plugin-specific knowledge in this file or in spi
//   - Adding a new native method = add it in Java + implement in Rust. Done.
//
// ═══════════════════════════════════════════════════════════════════════════════

// ── Global allocator ──
// All Rust code in this .so (from all plugin rlibs) shares this single mimalloc instance.
// This is why we need one .so — if there were two, each would have its own mimalloc,
// and freeing memory allocated by the other would segfault.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// ── Pull in plugin rlibs ──
// These `extern crate` statements force the linker to include all #[no_mangle] symbols
// from each plugin's Rust code into this cdylib. Without them, the linker would
// dead-strip the Java_*_* functions since nothing in this crate calls them directly.
// The symbols need to be present so dlsym() can find them at runtime.
extern crate opensearch_datafusion;
extern crate opensearch_parquet_format;

use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::{jint, JNI_VERSION_1_8};
use jni::JavaVM;
use std::os::raw::c_void;
use vectorized_exec_spi::native_registry::{NATIVE_REGISTRARS, get_class_name, register_natives_for_class};

/// Called by JVM immediately when the .so is loaded via System.load().
/// We just return the JNI version — no initialization needed here.
/// The actual native method binding happens later when each plugin calls registerClass().
#[unsafe(no_mangle)]
pub extern "system" fn JNI_OnLoad(_vm: JavaVM, _reserved: *mut c_void) -> jint {
    JNI_VERSION_1_8
}

/// The ONE native method that JVM can find through normal classloader lookup,
/// because NativeLibraryLoader.java is in the parent classloader (spi) — the same
/// classloader that loaded this .so.
///
/// Called from Java: `NativeLibraryLoader.registerClassNative(NativeBridge.class)`
///
/// What happens:
///   1. Gets the class name from the Class<?> object (e.g. "org.opensearch.datafusion.jni.NativeBridge")
///   2. Checks that this class was registered via register_native_bridge!() macro at compile time
///      (this is a safety check — the NATIVE_REGISTRARS slice is populated at link time by each plugin)
///   3. Delegates to register_natives_for_class() which:
///      a. Reflects over the Java class to find all `native` methods
///      b. For each native method, builds the JNI symbol name (e.g. "Java_org_opensearch_datafusion_jni_NativeBridge_getVersionInfo")
///      c. Looks up that symbol in the loaded .so via dlsym() — returns a function pointer
///      d. Calls JNI RegisterNatives to bind: "when Java calls getVersionInfo(), jump to this pointer"
///   4. After this, all native calls from that class go directly to the Rust functions — zero overhead
///
/// This runs ONCE per plugin at startup. After that, native calls are direct function pointer jumps.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_opensearch_vectorized_execution_jni_NativeLibraryLoader_registerClassNative<'a>(
    mut env: JNIEnv<'a>,
    _loader_class: JClass<'a>,
    target_class: JClass<'a>,
) {
    // Step 1: Get the fully qualified class name from the Class<?> object
    let class_name = match get_class_name(&mut env, &target_class) {
        Ok(n) => n,
        Err(e) => { let _ = env.throw_new("java/lang/RuntimeException", &e); return; }
    };

    // Step 2: Verify this class was registered by a plugin via register_native_bridge!()
    // NATIVE_REGISTRARS is a compile-time distributed slice — each plugin appends to it
    // via the macro. This is just a safety check (2-3 entries, constant time).
    if !NATIVE_REGISTRARS.iter().any(|r| r.class_name == class_name) {
        let _ = env.throw_new("java/lang/IllegalArgumentException",
            &format!("No native registrar for: {}. Did the plugin call register_native_bridge!()?", class_name));
        return;
    }

    // Step 3: Auto-discover native methods via reflection + dlsym, then RegisterNatives
    // See spi/src/native_registry.rs for the implementation
    if let Err(e) = register_natives_for_class(&mut env, &target_class) {
        let _ = env.throw_new("java/lang/RuntimeException", &e);
    }
}

/// Returns per-plugin heap stats as a JSON string.
/// Format: [{"name":"plugin-name","used":12345,"committed":65536}, ...]
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_opensearch_vectorized_execution_jni_NativeLibraryLoader_getHeapStatsNative<'a>(
    mut env: JNIEnv<'a>,
    _class: JClass<'a>,
) -> jni::sys::jstring {
    use vectorized_exec_spi::heap_allocator::all_plugin_stats;
    let stats = all_plugin_stats();
    let mut json = String::from("[");
    for (i, ps) in stats.iter().enumerate() {
        if i > 0 { json.push(','); }
        json.push_str(&format!(
            "{{\"name\":\"{}\",\"used\":{},\"committed\":{}}}",
            ps.name, ps.stats.used, ps.stats.committed
        ));
    }
    json.push(']');
    match env.new_string(&json) {
        Ok(s) => s.into_raw(),
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", format!("{}", e));
            std::ptr::null_mut()
        }
    }
}

/// Returns global mimalloc stats (aggregated over all heaps) as JSON.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_opensearch_vectorized_execution_jni_NativeLibraryLoader_getGlobalMimallocStatsNative<'a>(
    mut env: JNIEnv<'a>,
    _class: JClass<'a>,
) -> jni::sys::jstring {
    use vectorized_exec_spi::heap_allocator::global_mimalloc_stats_json;
    let json = global_mimalloc_stats_json();
    match env.new_string(&json) {
        Ok(s) => s.into_raw(),
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", format!("{}", e));
            std::ptr::null_mut()
        }
    }
}
