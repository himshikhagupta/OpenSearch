// ═══════════════════════════════════════════════════════════════════════════════
// FFM (Foreign Function & Memory) exports for engine-datafusion
// ═══════════════════════════════════════════════════════════════════════════════
//
// WHAT IS FFM?
//   Java's Foreign Function & Memory API (stable since Java 22) allows Java code
//   to call native functions directly — no JNI, no `native` keyword, no JNIEnv.
//   Java uses MethodHandle + FunctionDescriptor to describe the C function signature,
//   and Linker.downcallHandle() to create a callable handle.
//
// WHY FFM OVER JNI?
//   - No classloader issues: SymbolLookup.libraryLookup() is NOT tied to a classloader.
//     Any class in any classloader can use the same SymbolLookup. This eliminates the
//     entire RegisterNatives + dlsym + linkme SPI machinery we need for JNI.
//   - No JNI overhead: no JNIEnv pointer, no jobject handles, no local reference tables.
//   - Simpler Rust code: plain C-ABI functions with primitive types.
//   - Type-safe memory: Java's MemorySegment + Arena provide structured access to native
//     memory with automatic cleanup, replacing raw pointer juggling.
//
// HOW THESE FUNCTIONS DIFFER FROM JNI:
//   JNI function:
//     #[no_mangle]
//     pub extern "system" fn Java_org_opensearch_datafusion_jni_NativeBridge_getVersionInfo(
//         env: JNIEnv,        ← JVM passes this
//         class: JClass,      ← JVM passes this
//     ) -> JString { ... }    ← must create a JNI string via env.new_string()
//
//   FFM function (this file):
//     #[no_mangle]
//     pub extern "C" fn datafusion_get_version() -> *const c_char { ... }
//     ← no JNIEnv, no JClass, just a C string pointer
//
// HOW JAVA CALLS THESE:
//   // One-time setup (typically in a static block or constructor):
//   SymbolLookup lib = SymbolLookup.libraryLookup("opensearch_native_jni", Arena.global());
//   Linker linker = Linker.nativeLinker();
//
//   // Create a handle for each function:
//   MethodHandle getVersion = linker.downcallHandle(
//       lib.find("datafusion_get_version").orElseThrow(),
//       FunctionDescriptor.of(ValueLayout.ADDRESS)  // returns a pointer
//   );
//
//   // Call it:
//   MemorySegment ptr = (MemorySegment) getVersion.invokeExact();
//   String version = ptr.reinterpret(256).getString(0);
//
// NAMING CONVENTION:
//   FFM functions use: <module>_<function_name>  (e.g. datafusion_get_version)
//   JNI functions use: Java_<package>_<class>_<method> (e.g. Java_org_opensearch_..._getVersionInfo)
//   Both coexist in the same .so — different symbols, same binary.
//
// SINGLE MIMALLOC:
//   These functions run in the same .so as the JNI functions. All allocations
//   (Vec, Box, String, etc.) go through the single #[global_allocator] mimalloc
//   instance defined in jni-entry/src/lib.rs. Memory allocated by a parquet FFM
//   function can be safely freed by a datafusion FFM function, and vice versa.
//
// ═══════════════════════════════════════════════════════════════════════════════

use std::ffi::{c_char, c_long};

/// Returns the DataFusion version info as a null-terminated C string.
///
/// The returned pointer is to a static string — it lives for the entire process
/// lifetime and must NOT be freed by the caller.
///
/// # Java FFM usage
/// ```java
/// MethodHandle h = linker.downcallHandle(
///     lib.find("datafusion_get_version").orElseThrow(),
///     FunctionDescriptor.of(ValueLayout.ADDRESS)
/// );
/// MemorySegment ptr = (MemorySegment) h.invokeExact();
/// // reinterpret(256) because returned pointer has zero-length segment by default
/// String version = ptr.reinterpret(256).getString(0);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn datafusion_get_version() -> *const c_char {
    // Static byte array with null terminator — no allocation, no cleanup needed
    static VERSION: &[u8] = b"{\"version\": \"52.4.0\", \"codecs\": [\"CsvDataSourceCodec\"]}\0";
    VERSION.as_ptr() as *const c_char
}

/// Initialize the tokio runtime manager with the given number of CPU threads.
///
/// Returns 0 on success, -1 on invalid input.
/// In production, this would call into RuntimeManager::new() — the same code
/// path that the JNI version uses.
///
/// # Java FFM usage
/// ```java
/// MethodHandle h = linker.downcallHandle(
///     lib.find("datafusion_init_runtime").orElseThrow(),
///     FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG)
/// );
/// long status = (long) h.invokeExact(4L);  // 4 CPU threads
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn datafusion_init_runtime(cpu_threads: c_long) -> c_long {
    if cpu_threads <= 0 { return -1; }
    // TODO: wire to RuntimeManager::with_config() when migrating from JNI to FFM
    0
}

/// Allocate a buffer of `size` bytes via mimalloc, filled with 0xDF.
///
/// Returns a raw pointer to the allocated memory. The caller MUST free it
/// by calling datafusion_free_buffer(ptr, size) — or parquet_free_buffer(),
/// since they share the same mimalloc instance.
///
/// This function exists primarily to prove cross-module alloc/free works
/// with a single mimalloc. In production, you'd pass structured data
/// via MemorySegment layouts instead of raw buffers.
///
/// # Java FFM usage
/// ```java
/// MethodHandle alloc = linker.downcallHandle(
///     lib.find("datafusion_allocate_buffer").orElseThrow(),
///     FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
/// );
/// MemorySegment ptr = (MemorySegment) alloc.invokeExact(1024L);
/// // ptr now points to 1024 bytes of 0xDF
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn datafusion_allocate_buffer(size: c_long) -> *mut u8 {
    let buf = vec![0xDFu8; size as usize];
    Box::into_raw(buf.into_boxed_slice()) as *mut u8
}

/// Free a buffer previously allocated by datafusion_allocate_buffer()
/// OR parquet_allocate_buffer() — both use the same mimalloc instance.
///
/// # Safety
/// - `ptr` must have been returned by datafusion_allocate_buffer() or parquet_allocate_buffer()
/// - `size` must match the size passed to the allocate call
/// - Must not be called twice on the same pointer (double-free)
///
/// # Java FFM usage
/// ```java
/// MethodHandle free = linker.downcallHandle(
///     lib.find("datafusion_free_buffer").orElseThrow(),
///     FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
/// );
/// free.invokeExact(ptr, 1024L);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn datafusion_free_buffer(ptr: *mut u8, size: c_long) {
    if ptr.is_null() { return; }
    unsafe {
        let slice = std::slice::from_raw_parts_mut(ptr, size as usize);
        let _ = Box::from_raw(slice as *mut [u8]);
    }
}
