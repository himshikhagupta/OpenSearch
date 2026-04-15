/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.vectorized.execution.jni;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

/**
 * Loads the unified OpenSearch native JNI library and provides
 * {@link #registerClass(Class)} to bind native methods in plugin/module
 * classloaders to the single shared .so.
 *
 * <p>This class lives in vectorized-exec-spi (parent classloader).
 * The .so is loaded here once. Plugins call {@code registerClass(MyBridge.class)}
 * from their static initializer — Rust then calls {@code RegisterNatives} to
 * bind the native method declarations in that class to the correct function pointers.
 */
public final class NativeLibraryLoader {

    private static final String LIBRARY_NAME = "opensearch_native_jni";
    private static final String DEFAULT_PATH = "native";
    private static final Logger logger = LogManager.getLogger(NativeLibraryLoader.class);
    private static volatile boolean loaded = false;

    private NativeLibraryLoader() {}

    /**
     * Ensure the native library is loaded, then register native methods
     * for the given bridge class. Safe to call from multiple plugins —
     * the library is loaded only once.
     *
     * @param clazz the class containing {@code native} method declarations
     */
    public static synchronized void registerClass(Class<?> clazz) {
        ensureLoaded();
        registerClassNative(clazz);
    }

    /**
     * Load the library from an explicit absolute path. Used for testing.
     */
    public static synchronized void loadFromPath(String absolutePath) {
        if (loaded) return;
        System.load(absolutePath);
        loaded = true;
    }

    private static void ensureLoaded() {
        if (loaded) return;

        try {
            System.loadLibrary(LIBRARY_NAME);
            loaded = true;
            return;
        } catch (UnsatisfiedLinkError ignored) {
            logger.debug("Library '{}' not found on system path, trying resources", LIBRARY_NAME);
        }

        try {
            loadFromResources();
            return;
        } catch (UnsatisfiedLinkError | IOException e) {
            logger.debug("Library '{}' not found in default resources path", LIBRARY_NAME);
        }

        try {
            String platformDir = PlatformHelper.getPlatformDirectory();
            String libFile = PlatformHelper.getPlatformLibraryName(LIBRARY_NAME);
            Path path = Paths.get(System.getProperty("user.dir"), "native", platformDir, libFile);
            System.load(path.toString());
            loaded = true;
        } catch (UnsatisfiedLinkError e) {
            throw new NativeLoaderException(
                "Failed to load native library '" + LIBRARY_NAME + "' from all attempted locations", e
            );
        }
    }

    private static void loadFromResources() throws IOException {
        String platformDir = PlatformHelper.getPlatformDirectory();
        String libFile = PlatformHelper.getPlatformLibraryName(LIBRARY_NAME);
        String resourcePath = Paths.get("/", DEFAULT_PATH, platformDir, libFile).toString();

        try (InputStream is = NativeLibraryLoader.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new IOException("Native library not found in resources: " + resourcePath);
            }
            Path tempFile = Files.createTempFile(LIBRARY_NAME, PlatformHelper.getNativeExtension());
            tempFile.toFile().deleteOnExit();
            Files.copy(is, tempFile, StandardCopyOption.REPLACE_EXISTING);
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try { Files.deleteIfExists(tempFile); } catch (IOException ignored) {}
            }));
            System.load(tempFile.toAbsolutePath().toString());
            loaded = true;
        }
    }

    /**
     * Native method implemented in Rust. Receives a Class object, inspects its name,
     * and calls JNI RegisterNatives to bind the native methods in that class to
     * the Rust function pointers in the .so.
     */
    private static native void registerClassNative(Class<?> clazz);

    /**
     * Returns per-plugin native heap stats as a JSON string.
     * Format: [{"name":"plugin-name","used":12345,"committed":65536}, ...]
     * @return JSON array of per-plugin heap stats, or "[]" if no plugins registered
     */
    public static String getHeapStats() {
        ensureLoaded();
        return getHeapStatsNative();
    }

    private static native String getHeapStatsNative();

    /**
     * Returns global mimalloc process-level memory stats as a JSON string.
     * Format: {"current_rss":N,"peak_rss":N,"current_commit":N,"peak_commit":N}
     * @return JSON object with global mimalloc stats
     */
    public static String getGlobalMimallocStats() {
        ensureLoaded();
        return getGlobalMimallocStatsNative();
    }

    private static native String getGlobalMimallocStatsNative();
}
