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
 *   1. Build the Rust native library:
 *        cd libs/vectorized-exec-spi/rust
 *        cargo build -p opensearch-native-jni
 *
 *   2. Build the Java project (to get compiled bridge classes):
 *        ./gradlew :plugins:engine-datafusion:compileJava
 *        ./gradlew :modules:parquet-data-format:compileJava
 *
 *   3. Create stub jars for the two bridge classes (simulating child classloaders).
 *      These must have the exact same native method signatures as the real classes
 *      but WITHOUT static initializer blocks (no NativeLibraryLoader.load() calls):
 *
 *        # From the repo root:
 *        mkdir -p /tmp/jni-test/{child1,child2}
 *
 *        # Copy compiled classes (includes all dependencies like ActionListener):
 *        cp -r plugins/engine-datafusion/build/classes/java/main/org /tmp/jni-test/child1/
 *        cp -r libs/core/build/classes/java/main/org /tmp/jni-test/child1/
 *        cd /tmp/jni-test/child1 && jar cf ../child1.jar org/ && cd -
 *
 *        cp -r modules/parquet-data-format/build/classes/java/main/com /tmp/jni-test/child2/
 *        cd /tmp/jni-test/child2 && jar cf ../child2.jar com/ && cd -
 *
 *      NOTE: The child jars must contain stub bridge classes WITHOUT static {} blocks,
 *      or the real classes will try to call NativeLibraryLoader.load() which conflicts
 *      with this test's controlled loading sequence.
 *
 *   4. Compile this test:
 *        javac -cp <log4j-api.jar> \
 *          libs/vectorized-exec-spi/src/main/java/org/opensearch/vectorized/execution/jni/PlatformHelper.java \
 *          libs/vectorized-exec-spi/src/main/java/org/opensearch/vectorized/execution/jni/NativeLoaderException.java \
 *          libs/vectorized-exec-spi/src/main/java/org/opensearch/vectorized/execution/jni/NativeLibraryLoader.java \
 *          libs/vectorized-exec-spi/src/test/java/org/opensearch/vectorized/execution/jni/RegisterNativesProof.java \
 *          -d /tmp/jni-test/classes
 *
 *   5. Run (Java 21+):
 *        java -cp /tmp/jni-test/classes:<log4j-api.jar> \
 *          -Dnative.lib.path=$(pwd)/libs/vectorized-exec-spi/rust/target/debug/libopensearch_native_jni.dylib \
 *          -Dchild1.jar=/tmp/jni-test/child1.jar \
 *          -Dchild2.jar=/tmp/jni-test/child2.jar \
 *          org.opensearch.vectorized.execution.jni.RegisterNativesProof
 *
 * Expected output:
 *   [PASS] NativeBridge.getVersionInfo() from child classloader
 *   [PASS] RustBridge.getFilteredNativeBytesUsed() from child classloader
 *   [PASS] DataFusion freed parquet's buffer — single mimalloc confirmed
 *   === Results: 3 passed, 0 failed ===
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

package org.opensearch.vectorized.execution.jni;

import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Paths;

/**
 * Proves:
 * 1. RegisterNatives works across classloaders (native methods in child CLs, .so in parent)
 * 2. Single mimalloc instance (alloc in parquet module, free in datafusion plugin — no crash)
 */
public class RegisterNativesProof {

    public static void main(String[] args) throws Exception {
        String dylib = System.getProperty("native.lib.path");
        String child1JarPath = System.getProperty("child1.jar");
        String child2JarPath = System.getProperty("child2.jar");

        System.out.println("=== Single .so Cross-ClassLoader Proof ===\n");

        // Load .so in parent classloader
        NativeLibraryLoader.loadFromPath(dylib);
        ClassLoader parent = RegisterNativesProof.class.getClassLoader();
        System.out.println("[PARENT] .so loaded in: " + parent + "\n");

        // Create isolated child classloaders (simulating plugin/module)
        URL jar1 = Paths.get(child1JarPath).toUri().toURL();
        URL jar2 = Paths.get(child2JarPath).toUri().toURL();
        URLClassLoader cl1 = new URLClassLoader("plugin-datafusion", new URL[]{jar1}, parent);
        URLClassLoader cl2 = new URLClassLoader("module-parquet", new URL[]{jar2}, parent);

        // Load bridge classes from child classloaders
        Class<?> nativeBridge = cl1.loadClass("org.opensearch.datafusion.jni.NativeBridge");
        Class<?> rustBridge = cl2.loadClass("com.parquet.parquetdataformat.bridge.RustBridge");
        System.out.println("[CHILD1] NativeBridge loaded by: " + nativeBridge.getClassLoader());
        System.out.println("[CHILD2] RustBridge   loaded by: " + rustBridge.getClassLoader() + "\n");

        // Register natives via SPI
        System.out.println("[REGISTER] Binding NativeBridge natives...");
        NativeLibraryLoader.registerClass(nativeBridge);
        System.out.println("[REGISTER] Binding RustBridge natives...\n");
        NativeLibraryLoader.registerClass(rustBridge);

        int passed = 0, failed = 0;

        // ── Test 1: DataFusion native method works from child CL ──
        System.out.println("[TEST 1] NativeBridge.getVersionInfo() from child classloader");
        try {
            Object result = nativeBridge.getMethod("getVersionInfo").invoke(null);
            System.out.println("[PASS]   " + result);
            passed++;
        } catch (Exception e) {
            Throwable c = e.getCause() != null ? e.getCause() : e;
            System.out.println("[FAIL]   " + c.getClass().getSimpleName() + ": " + c.getMessage());
            failed++;
        }

        // ── Test 2: Parquet native method works from child CL ──
        System.out.println("[TEST 2] RustBridge.getFilteredNativeBytesUsed() from child classloader");
        try {
            Object result = rustBridge.getMethod("getFilteredNativeBytesUsed", String.class).invoke(null, "/nonexistent");
            System.out.println("[PASS]   " + result);
            passed++;
        } catch (Exception e) {
            Throwable c = e.getCause() != null ? e.getCause() : e;
            System.out.println("[FAIL]   " + c.getClass().getSimpleName() + ": " + c.getMessage());
            failed++;
        }

        // ── Test 3: Cross-module alloc/free (single mimalloc proof) ──
        System.out.println("[TEST 3] Alloc 1MB in parquet, free in datafusion (single mimalloc proof)");
        try {
            long size = 1024 * 1024;
            Method alloc = rustBridge.getMethod("allocateTestBuffer", long.class);
            long ptr = (long) alloc.invoke(null, size);
            System.out.println("         Parquet allocated at: 0x" + Long.toHexString(ptr));

            Method free = nativeBridge.getMethod("freeTestBuffer", long.class, long.class);
            free.invoke(null, ptr, size);
            System.out.println("[PASS]   DataFusion freed it — no crash, single mimalloc confirmed");
            passed++;
        } catch (Exception e) {
            Throwable c = e.getCause() != null ? e.getCause() : e;
            System.out.println("[FAIL]   " + c.getClass().getSimpleName() + ": " + c.getMessage());
            failed++;
        }

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        cl1.close();
        cl2.close();
        if (failed > 0) System.exit(1);
    }
}
