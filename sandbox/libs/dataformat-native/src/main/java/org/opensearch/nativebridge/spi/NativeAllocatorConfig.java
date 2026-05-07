/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.nativebridge.spi;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * Dynamic jemalloc tuning via FFM.
 * <p>
 * Provides methods to adjust jemalloc's {@code dirty_decay_ms} and {@code muzzy_decay_ms}
 * at runtime for all arenas. These are called by plugin-level cluster settings listeners.
 * <p>
 * Note: {@code lg_tcache_max} is NOT dynamically tunable by jemalloc (init-time only).
 */
public final class NativeAllocatorConfig {

    private static final Logger logger = LogManager.getLogger(NativeAllocatorConfig.class);

    private static final MethodHandle SET_DIRTY;
    private static final MethodHandle SET_MUZZY;
    private static final MethodHandle START_MONITOR;
    private static final MethodHandle STOP_MONITOR;

    static {
        SymbolLookup lookup = NativeLibraryLoader.symbolLookup();
        Linker linker = Linker.nativeLinker();
        FunctionDescriptor desc = FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG);
        SET_DIRTY = linker.downcallHandle(lookup.find("native_jemalloc_set_dirty_decay_ms").orElseThrow(), desc);
        SET_MUZZY = linker.downcallHandle(lookup.find("native_jemalloc_set_muzzy_decay_ms").orElseThrow(), desc);
        START_MONITOR = linker.downcallHandle(lookup.find("native_start_plugin_monitor").orElseThrow(), desc);
        STOP_MONITOR = linker.downcallHandle(
            lookup.find("native_stop_plugin_monitor").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_LONG)
        );
    }

    private NativeAllocatorConfig() {}

    /**
     * Sets dirty_decay_ms for all jemalloc arenas. No restart required.
     *
     * @param ms decay time in milliseconds (-1 to disable decay)
     */
    public static void setDirtyDecayMs(long ms) {
        applyDecay(SET_DIRTY, "dirty_decay_ms", ms);
    }

    /**
     * Sets muzzy_decay_ms for all jemalloc arenas. No restart required.
     *
     * @param ms decay time in milliseconds (-1 to disable decay)
     */
    public static void setMuzzyDecayMs(long ms) {
        applyDecay(SET_MUZZY, "muzzy_decay_ms", ms);
    }

    private static void applyDecay(MethodHandle handle, String name, long ms) {
        try {
            long rc = (long) handle.invokeExact(ms);
            NativeLibraryLoader.checkResult(rc);
            logger.info("jemalloc {} updated to {}", name, ms);
        } catch (Throwable t) {
            logger.warn("Error setting jemalloc " + name, t);
        }
    }

    /**
     * Starts the native memory monitoring thread at the given interval.
     *
     * @param intervalSecs logging interval in seconds
     */
    public static void startMonitor(long intervalSecs) {
        try {
            long rc = (long) START_MONITOR.invokeExact(intervalSecs);
            NativeLibraryLoader.checkResult(rc);
            logger.info("Native memory monitor started (interval={}s)", intervalSecs);
        } catch (Throwable t) {
            logger.warn("Error starting native memory monitor", t);
        }
    }

    /**
     * Stops the native memory monitoring thread.
     */
    public static void stopMonitor() {
        try {
            long rc = (long) STOP_MONITOR.invokeExact();
            NativeLibraryLoader.checkResult(rc);
            logger.info("Native memory monitor stopped");
        } catch (Throwable t) {
            logger.warn("Error stopping native memory monitor", t);
        }
    }
}
