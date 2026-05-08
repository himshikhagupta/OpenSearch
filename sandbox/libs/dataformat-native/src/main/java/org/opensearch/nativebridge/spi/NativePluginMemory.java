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
 * Per-plugin native memory tracking via jemalloc arena groups.
 * <p>
 * Each plugin registers to get a dedicated set of jemalloc arenas.
 * Threads are bound to a plugin's arena group so all allocations
 * on that thread are attributed to the plugin. jemalloc handles
 * cross-plugin frees correctly via chunk metadata.
 */
public final class NativePluginMemory {

    private static final Logger logger = LogManager.getLogger(NativePluginMemory.class);

    private static final MethodHandle REGISTER;
    private static final MethodHandle BIND_THREAD;
    private static final MethodHandle ALLOCATED_BYTES;
    private static final MethodHandle PLUGIN_COUNT;

    static {
        SymbolLookup lookup = NativeLibraryLoader.symbolLookup();
        Linker linker = Linker.nativeLinker();

        REGISTER = linker.downcallHandle(
            lookup.find("native_plugin_register").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_LONG)
        );
        BIND_THREAD = linker.downcallHandle(
            lookup.find("native_plugin_bind_thread").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG)
        );
        ALLOCATED_BYTES = linker.downcallHandle(
            lookup.find("native_plugin_allocated_bytes").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG)
        );
        PLUGIN_COUNT = linker.downcallHandle(
            lookup.find("native_plugin_count").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_LONG)
        );
    }

    private NativePluginMemory() {}

    /**
     * Register a new plugin. Returns the plugin_id (1-based).
     * Call once per plugin during initialization.
     */
    public static long register() {
        try {
            long rc = (long) REGISTER.invokeExact();
            return NativeLibraryLoader.checkResult(rc);
        } catch (Throwable t) {
            throw new RuntimeException("native_plugin_register failed", t);
        }
    }

    /**
     * Bind the calling thread to a plugin's arena group.
     * All subsequent native allocations on this thread will be attributed to the plugin.
     *
     * @param pluginId the plugin_id returned by {@link #register()}, or 0 to unbind
     */
    public static void bindThread(long pluginId) {
        try {
            long rc = (long) BIND_THREAD.invokeExact(pluginId);
            NativeLibraryLoader.checkResult(rc);
        } catch (Throwable t) {
            logger.warn("native_plugin_bind_thread failed for plugin {}", pluginId, t);
        }
    }

    /**
     * Get the current allocated bytes for a plugin.
     *
     * @param pluginId the plugin_id returned by {@link #register()}
     * @return allocated bytes, or -1 on error
     */
    public static long allocatedBytes(long pluginId) {
        try {
            long rc = (long) ALLOCATED_BYTES.invokeExact(pluginId);
            return NativeLibraryLoader.checkResult(rc);
        } catch (Throwable t) {
            logger.warn("native_plugin_allocated_bytes failed for plugin {}", pluginId, t);
            return -1;
        }
    }

    /**
     * Get the number of registered plugins.
     */
    public static long pluginCount() {
        try {
            long rc = (long) PLUGIN_COUNT.invokeExact();
            return NativeLibraryLoader.checkResult(rc);
        } catch (Throwable t) {
            logger.warn("native_plugin_count failed", t);
            return -1;
        }
    }
}
