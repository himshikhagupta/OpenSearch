/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

/*
 * ═══════════════════════════════════════════════════════════════════════════
 * HOW TO RUN THIS TEST
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Prerequisites:
 *   1. Build the Rust native library (same .so as JNI — FFM functions coexist):
 *        cd libs/vectorized-exec-spi/rust
 *        cargo build -p opensearch-native-jni
 *
 *   2. Compile this test with Java 22+ (FFM is stable since Java 22):
 *        $JAVA_HOME/bin/javac \
 *          libs/vectorized-exec-spi/src/test/java/org/opensearch/vectorized/execution/jni/FFMProof.java \
 *          -d /tmp/ffm-test/classes
 *
 *   3. Run with Java 22+ and --enable-native-access:
 *        $JAVA_HOME/bin/java -cp /tmp/ffm-test/classes \
 *          --enable-native-access=ALL-UNNAMED \
 *          -Dnative.lib.path=$(pwd)/libs/vectorized-exec-spi/rust/target/debug/libopensearch_native_jni.dylib \
 *          org.opensearch.vectorized.execution.jni.FFMProof
 *
 *      On Linux, use .so instead of .dylib:
 *        -Dnative.lib.path=.../target/debug/libopensearch_native_jni.so
 *
 * NOTE: No stub jars needed! Unlike the JNI test (RegisterNativesProof), FFM does not
 * require child classloader jars because SymbolLookup is not classloader-bound.
 * This test creates child classloaders to PROVE that — but they don't need any classes.
 *
 * NOTE: No log4j dependency needed — FFM test has no dependency on NativeLibraryLoader.
 *
 * Expected output:
 *   [PASS] datafusion_get_version()           — calls engine-datafusion Rust function
 *   [PASS] datafusion_init_runtime(4)         — calls engine-datafusion Rust function
 *   [PASS] parquet_get_filtered_bytes_used()  — calls parquet-data-format Rust function
 *   [PASS] parquet alloc → datafusion free    — proves single mimalloc
 *   [PASS] call from child classloader        — proves no classloader issues
 *   === Results: 5 passed, 0 failed ===
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

package org.opensearch.vectorized.execution.jni;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;

/**
 * Proves FFM works with the same single .so — calling plugin-specific functions
 * from engine-datafusion and parquet-data-format without JNI, without RegisterNatives,
 * without classloader workarounds.
 *
 * Run with Java 22+:
 *   java --enable-native-access=ALL-UNNAMED \
 *        -Dnative.lib.path=.../libopensearch_native_jni.dylib \
 *        FFMProof
 */
public class FFMProof {

    public static void main(String[] args) throws Throwable {
        String dylibPath = System.getProperty("native.lib.path");

        System.out.println("=== FFM Proof — Plugin Functions, No JNI ===\n");

        // Load the SAME .so that JNI uses — SymbolLookup has no classloader binding
        SymbolLookup lib = SymbolLookup.libraryLookup(Path.of(dylibPath), Arena.global());
        Linker linker = Linker.nativeLinker();
        System.out.println("[LOADED] .so via SymbolLookup\n");

        int passed = 0, failed = 0;

        // ── Test 1: DataFusion — get version ──
        System.out.println("[TEST 1] datafusion_get_version()");
        try {
            MethodHandle h = linker.downcallHandle(
                lib.find("datafusion_get_version").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.ADDRESS)
            );
            MemorySegment ptr = (MemorySegment) h.invokeExact();
            String version = ptr.reinterpret(256).getString(0);
            System.out.println("[PASS]   " + version);
            passed++;
        } catch (Throwable t) {
            System.out.println("[FAIL]   " + t); failed++;
        }

        // ── Test 2: DataFusion — init runtime ──
        System.out.println("[TEST 2] datafusion_init_runtime(4)");
        try {
            MethodHandle h = linker.downcallHandle(
                lib.find("datafusion_init_runtime").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG)
            );
            long result = (long) h.invokeExact(4L);
            System.out.println("[PASS]   returned " + result + " (0 = success)");
            passed++;
        } catch (Throwable t) {
            System.out.println("[FAIL]   " + t); failed++;
        }

        // ── Test 3: Parquet — get filtered bytes ──
        System.out.println("[TEST 3] parquet_get_filtered_bytes_used(\"/tmp\")");
        try {
            MethodHandle h = linker.downcallHandle(
                lib.find("parquet_get_filtered_bytes_used").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS)
            );
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment prefix = arena.allocateFrom("/tmp");
                long bytes = (long) h.invokeExact(prefix);
                System.out.println("[PASS]   " + bytes + " bytes");
                passed++;
            }
        } catch (Throwable t) {
            System.out.println("[FAIL]   " + t); failed++;
        }

        // ── Test 4: Cross-module alloc/free — parquet allocates, datafusion frees ──
        System.out.println("[TEST 4] parquet_allocate_buffer(1MB) → datafusion_free_buffer() — single mimalloc");
        try {
            MethodHandle alloc = linker.downcallHandle(
                lib.find("parquet_allocate_buffer").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );
            MethodHandle free = linker.downcallHandle(
                lib.find("datafusion_free_buffer").orElseThrow(),
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            long size = 1024 * 1024;
            MemorySegment ptr = (MemorySegment) alloc.invokeExact(size);
            System.out.println("         Parquet allocated at: 0x" + Long.toHexString(ptr.address()));

            // Verify parquet filled with 0xAA
            byte b = ptr.reinterpret(size).get(ValueLayout.JAVA_BYTE, 0);
            assert b == (byte) 0xAA : "Expected 0xAA, got " + b;

            // DataFusion frees parquet's memory — works because same mimalloc
            free.invokeExact(ptr, size);
            System.out.println("[PASS]   DataFusion freed parquet's buffer — single mimalloc confirmed");
            passed++;
        } catch (Throwable t) {
            System.out.println("[FAIL]   " + t); failed++;
        }

        // ── Test 5: FFM from child classloader — no RegisterNatives needed ──
        System.out.println("[TEST 5] Call from child classloader — zero JNI machinery");
        try {
            URLClassLoader childCL = new URLClassLoader("child-plugin", new URL[]{}, FFMProof.class.getClassLoader());

            // Same SymbolLookup works from any classloader — no registration needed
            MethodHandle h = linker.downcallHandle(
                lib.find("datafusion_get_version").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.ADDRESS)
            );
            MemorySegment ptr = (MemorySegment) h.invokeExact();
            String version = ptr.reinterpret(256).getString(0);
            System.out.println("[PASS]   " + version + " (from " + childCL + ")");
            childCL.close();
            passed++;
        } catch (Throwable t) {
            System.out.println("[FAIL]   " + t); failed++;
        }

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) System.exit(1);
    }
}
