/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! FFM bridge for the Parquet writer.
//!
//! Return convention: `>= 0` success, `< 0` error pointer (negate to get ptr,
//! call `native_error_message`/`native_error_free`).

use std::slice;
use std::str;

use native_bridge_common::ffm_safe;
use native_bridge_common::heap_allocator::{scoped_thread_heap, HeapGuard, PluginHeap};

use crate::writer::NativeParquetWriter;

use std::sync::OnceLock;

static PLUGIN_HEAP: OnceLock<PluginHeap> = OnceLock::new();

fn scoped_plugin_heap() -> Option<HeapGuard> {
    PLUGIN_HEAP.get().map(|h| scoped_thread_heap(*h))
}

unsafe fn str_from_raw<'a>(ptr: *const u8, len: i64) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err("null string pointer".to_string());
    }
    if len < 0 {
        return Err(format!("negative string length: {}", len));
    }
    let bytes = slice::from_raw_parts(ptr, len as usize);
    str::from_utf8(bytes).map_err(|e| format!("invalid UTF-8: {}", e))
}

#[ffm_safe]
#[no_mangle]
pub unsafe extern "C" fn parquet_create_writer(
    file_ptr: *const u8,
    file_len: i64,
    schema_address: i64,
) -> i64 {
    let _guard = scoped_plugin_heap();
    let filename = str_from_raw(file_ptr, file_len).map_err(|e| format!("parquet_create_writer: {}", e))?.to_string();
    NativeParquetWriter::create_writer(filename, schema_address)
        .map(|_| 0)
        .map_err(|e| e.to_string())
}

#[ffm_safe]
#[no_mangle]
pub unsafe extern "C" fn parquet_write(
    file_ptr: *const u8,
    file_len: i64,
    array_address: i64,
    schema_address: i64,
) -> i64 {
    let _guard = scoped_plugin_heap();
    let filename = str_from_raw(file_ptr, file_len).map_err(|e| format!("parquet_write: {}", e))?.to_string();
    NativeParquetWriter::write_data(filename, array_address, schema_address)
        .map(|_| 0)
        .map_err(|e| e.to_string())
}

/// Returns 0 with metadata in out-pointers, 1 if no writer found.
#[ffm_safe]
#[no_mangle]
pub unsafe extern "C" fn parquet_finalize_writer(
    file_ptr: *const u8,
    file_len: i64,
    version_out: *mut i32,
    num_rows_out: *mut i64,
    created_by_buf: *mut u8,
    created_by_buf_len: i64,
    created_by_len_out: *mut i64,
    crc32_out: *mut i64,
) -> i64 {
    let filename = str_from_raw(file_ptr, file_len).map_err(|e| format!("parquet_finalize_writer: {}", e))?.to_string();
    let _guard = scoped_plugin_heap();
    match NativeParquetWriter::finalize_writer(filename) {
        Ok(Some(result)) => {
            let fm = result.metadata.file_metadata();
            if !version_out.is_null() { *version_out = fm.version(); }
            if !num_rows_out.is_null() { *num_rows_out = fm.num_rows(); }
            if let Some(cb) = fm.created_by() {
                if !created_by_buf.is_null() && created_by_buf_len > 0 {
                    let bytes = cb.as_bytes();
                    let n = bytes.len().min(created_by_buf_len as usize);
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), created_by_buf, n);
                    if !created_by_len_out.is_null() { *created_by_len_out = n as i64; }
                }
            } else if !created_by_len_out.is_null() {
                *created_by_len_out = -1;
            }
            if !crc32_out.is_null() { *crc32_out = result.crc32 as i64; }
            Ok(0)
        }
        Ok(None) => Ok(1),
        Err(e) => Err(e.to_string()),
    }
}

#[ffm_safe]
#[no_mangle]
pub unsafe extern "C" fn parquet_sync_to_disk(
    file_ptr: *const u8,
    file_len: i64,
) -> i64 {
    let _guard = scoped_plugin_heap();
    let filename = str_from_raw(file_ptr, file_len).map_err(|e| format!("parquet_sync_to_disk: {}", e))?.to_string();
    NativeParquetWriter::sync_to_disk(filename)
        .map(|_| 0)
        .map_err(|e| e.to_string())
}

#[ffm_safe]
#[no_mangle]
pub unsafe extern "C" fn parquet_get_file_metadata(
    file_ptr: *const u8,
    file_len: i64,
    version_out: *mut i32,
    num_rows_out: *mut i64,
    created_by_buf: *mut u8,
    created_by_buf_len: i64,
    created_by_len_out: *mut i64,
) -> i64 {
    let filename = str_from_raw(file_ptr, file_len).map_err(|e| format!("parquet_get_file_metadata: {}", e))?.to_string();
    let _guard = scoped_plugin_heap();
    let fm = NativeParquetWriter::get_file_metadata(filename).map_err(|e| e.to_string())?;
    if !version_out.is_null() { *version_out = fm.version(); }
    if !num_rows_out.is_null() { *num_rows_out = fm.num_rows(); }
    if let Some(cb) = fm.created_by() {
        if !created_by_buf.is_null() && created_by_buf_len > 0 {
            let bytes = cb.as_bytes();
            let n = bytes.len().min(created_by_buf_len as usize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), created_by_buf, n);
            if !created_by_len_out.is_null() { *created_by_len_out = n as i64; }
        }
    } else if !created_by_len_out.is_null() {
        *created_by_len_out = -1;
    }
    Ok(0)
}

#[no_mangle]
pub unsafe extern "C" fn parquet_get_filtered_native_bytes_used(
    prefix_ptr: *const u8,
    prefix_len: i64,
) -> i64 {
    let _guard = scoped_plugin_heap();
    let prefix = str_from_raw(prefix_ptr, prefix_len).unwrap_or("").to_string();
    NativeParquetWriter::get_filtered_writer_memory_usage(prefix).unwrap_or(0) as i64
}

/// Initialize the parquet plugin heap. Call once from Java before any parquet operations.
#[no_mangle]
pub extern "C" fn parquet_init_heap() {
    let heap = native_bridge_common::heap_allocator::create_heap("parquet");
    let _ = PLUGIN_HEAP.set(heap);
}

/// Allocate `size` bytes on the parquet plugin heap and return the pointer as i64.
/// Used by integration tests to verify heap tracking. Free with `parquet_free_test_buffer`.
#[no_mangle]
pub extern "C" fn parquet_allocate_test_buffer(size: i64) -> i64 {
    let _guard = scoped_plugin_heap();
    let buf: Vec<u8> = vec![0u8; size as usize];
    let ptr = buf.as_ptr() as i64;
    std::mem::forget(buf);
    ptr
}

/// Free a buffer previously allocated by `parquet_allocate_test_buffer`.
#[no_mangle]
pub unsafe extern "C" fn parquet_free_test_buffer(ptr: i64, size: i64) {
    let _guard = scoped_plugin_heap();
    let _ = Vec::from_raw_parts(ptr as *mut u8, size as usize, size as usize);
}
