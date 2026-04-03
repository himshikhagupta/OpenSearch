// ═══════════════════════════════════════════════════════════════════════════════
// FFM (Foreign Function & Memory) exports for parquet-data-format
// ═══════════════════════════════════════════════════════════════════════════════
//
// Plain C-ABI functions callable from Java 22+ via the FFM API.
// No JNIEnv, no JClass, no jni crate — just primitive types and pointers.
//
// These coexist with the JNI functions (Java_com_parquet_..._*) in the same .so.
// Both paths share the single mimalloc allocator, so memory allocated via FFM
// can be freed via JNI and vice versa.
//
// NAMING CONVENTION:
//   parquet_<function_name>  — e.g. parquet_get_filtered_bytes_used
//   This avoids collision with the JNI names (Java_com_parquet_..._*) and makes
//   it clear which calling convention is expected.
//
// HOW JAVA CALLS THESE (one-time setup):
//   SymbolLookup lib = SymbolLookup.libraryLookup("opensearch_native_jni", Arena.global());
//   Linker linker = Linker.nativeLinker();
//
//   // No classloader issues — SymbolLookup works from any classloader.
//   // No RegisterNatives — Java resolves symbols directly by name.
//   // No reflection — FunctionDescriptor describes the C signature explicitly.
//
// CROSS-MODULE ALLOC/FREE:
//   parquet_allocate_buffer() allocates via mimalloc (fills with 0xAA).
//   datafusion_free_buffer() frees it — works because same mimalloc instance.
//   This is the key reason we need a single .so: separate .so files would have
//   separate mimalloc instances, and cross-library free would segfault.
//
// ═══════════════════════════════════════════════════════════════════════════════

use std::ffi::{c_char, c_long};

/// Returns the total memory used by parquet writers matching a path prefix.
///
/// Takes a null-terminated C string as the path prefix filter.
/// Returns the byte count as a long.
///
/// In production, this would call NativeParquetWriter::get_filtered_writer_memory_usage().
/// Currently returns 0 (no active writers in the test context).
///
/// # Java FFM usage
/// ```java
/// MethodHandle h = linker.downcallHandle(
///     lib.find("parquet_get_filtered_bytes_used").orElseThrow(),
///     FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS)
/// );
/// try (Arena arena = Arena.ofConfined()) {
///     MemorySegment prefix = arena.allocateFrom("/data/indices");
///     long bytes = (long) h.invokeExact(prefix);
/// }
/// // Arena.ofConfined() auto-frees the prefix string when the try block exits.
/// // This is safer than JNI where you'd manually manage JString references.
/// ```
///
/// # Comparison with JNI version
/// JNI:  fn(env: JNIEnv, class: JClass, prefix: JString) -> jlong
///       ← must call env.get_string(&prefix) to convert JString to Rust &str
/// FFM:  fn(prefix: *const c_char) -> c_long
///       ← already a C string, can use CStr::from_ptr() directly
#[unsafe(no_mangle)]
pub extern "C" fn parquet_get_filtered_bytes_used(_prefix: *const c_char) -> c_long {
    // TODO: wire to NativeParquetWriter::get_filtered_writer_memory_usage()
    // when migrating from JNI to FFM
    0
}

/// Allocate a buffer of `size` bytes via mimalloc, filled with 0xAA.
///
/// Returns a raw pointer. The caller MUST free it by calling either:
///   - parquet_free_buffer(ptr, size)
///   - datafusion_free_buffer(ptr, size)  ← works because same mimalloc
///
/// The 0xAA fill value is used in tests to verify which module allocated
/// the buffer (datafusion uses 0xDF). In production, you'd use structured
/// MemorySegment layouts with Arena-managed lifetimes instead.
///
/// # Java FFM usage
/// ```java
/// MethodHandle alloc = linker.downcallHandle(
///     lib.find("parquet_allocate_buffer").orElseThrow(),
///     FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
/// );
/// MemorySegment ptr = (MemorySegment) alloc.invokeExact(1048576L);  // 1MB
///
/// // Read the first byte to verify it's 0xAA (parquet's signature)
/// byte b = ptr.reinterpret(1048576L).get(ValueLayout.JAVA_BYTE, 0);
/// assert b == (byte) 0xAA;
///
/// // Free via datafusion — proves single mimalloc
/// MethodHandle free = linker.downcallHandle(
///     lib.find("datafusion_free_buffer").orElseThrow(),
///     FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
/// );
/// free.invokeExact(ptr, 1048576L);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn parquet_allocate_buffer(size: c_long) -> *mut u8 {
    let buf = vec![0xAAu8; size as usize];
    Box::into_raw(buf.into_boxed_slice()) as *mut u8
}

/// Free a buffer previously allocated by parquet_allocate_buffer()
/// OR datafusion_allocate_buffer() — both use the same mimalloc instance.
///
/// # Safety
/// - `ptr` must have been returned by parquet_allocate_buffer() or datafusion_allocate_buffer()
/// - `size` must match the size passed to the allocate call
/// - Must not be called twice on the same pointer (double-free)
///
/// # Java FFM usage
/// ```java
/// MethodHandle free = linker.downcallHandle(
///     lib.find("parquet_free_buffer").orElseThrow(),
///     FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
/// );
/// free.invokeExact(ptr, size);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn parquet_free_buffer(ptr: *mut u8, size: c_long) {
    if ptr.is_null() { return; }
    unsafe {
        let slice = std::slice::from_raw_parts_mut(ptr, size as usize);
        let _ = Box::from_raw(slice as *mut [u8]);
    }
}
