/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics;

import org.opensearch.be.datafusion.nativelib.NativeBridge;
import org.opensearch.nativebridge.spi.NativeLibraryLoader;
import org.opensearch.parquet.bridge.RustBridge;
import org.opensearch.plugins.Plugin;
import org.opensearch.test.OpenSearchIntegTestCase;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Integration test that verifies per-plugin mimalloc heap tracking.
 *
 * <p>Initializes both DataFusion and Parquet plugin heaps, allocates test
 * buffers on each, and asserts that {@link NativeLibraryLoader#heapCount()},
 * {@link NativeLibraryLoader#heapName(int)}, and {@link NativeLibraryLoader#heapUsed(int)}
 * report the correct per-plugin memory usage.</p>
 */
@OpenSearchIntegTestCase.ClusterScope(scope = OpenSearchIntegTestCase.Scope.SUITE, numDataNodes = 1)
public class HeapStatsIT extends OpenSearchIntegTestCase {

    @Override
    protected Collection<Class<? extends Plugin>> nodePlugins() {
        return List.of(AnalyticsPlugin.class);
    }

    public void testHeapTrackingAcrossPlugins() {
        // Init both plugin heaps
        NativeBridge.initTokioRuntimeManager(1); // also creates datafusion heap
        RustBridge.initHeap();

        // Verify at least 2 heaps registered
        int count = NativeLibraryLoader.heapCount();
        assertTrue("Expected at least 2 heaps, got " + count, count >= 2);

        // Collect registered heap names
        Set<String> names = new HashSet<>();
        for (int i = 0; i < count; i++) {
            String name = NativeLibraryLoader.heapName(i);
            assertNotNull("Heap name at index " + i + " should not be null", name);
            names.add(name);
        }
        assertTrue("Expected 'datafusion' heap, got " + names, names.contains("datafusion"));
        assertTrue("Expected 'parquet' heap, got " + names, names.contains("parquet"));

        // Find indices
        int dfIdx = -1;
        int pqIdx = -1;
        for (int i = 0; i < count; i++) {
            String n = NativeLibraryLoader.heapName(i);
            if ("datafusion".equals(n)) dfIdx = i;
            if ("parquet".equals(n)) pqIdx = i;
        }

        // Allocate 256KB on datafusion, 512KB on parquet
        long dfSize = 256 * 1024;
        long pqSize = 512 * 1024;
        long dfPtr = NativeBridge.allocateTestBuffer(dfSize);
        long pqPtr = RustBridge.allocateTestBuffer(pqSize);
        assertTrue("DataFusion test buffer should be non-zero", dfPtr != 0);
        assertTrue("Parquet test buffer should be non-zero", pqPtr != 0);

        // Check heap stats
        long dfUsed = NativeLibraryLoader.heapUsed(dfIdx);
        long pqUsed = NativeLibraryLoader.heapUsed(pqIdx);
        assertTrue("DataFusion used=" + dfUsed + " should be >= " + dfSize, dfUsed >= dfSize);
        assertTrue("Parquet used=" + pqUsed + " should be >= " + pqSize, pqUsed >= pqSize);
        assertTrue("Parquet used=" + pqUsed + " should be > DataFusion used=" + dfUsed, pqUsed > dfUsed);

        // Free and verify decrease
        NativeBridge.freeTestBuffer(dfPtr, dfSize);
        RustBridge.freeTestBuffer(pqPtr, pqSize);

        long dfAfter = NativeLibraryLoader.heapUsed(dfIdx);
        long pqAfter = NativeLibraryLoader.heapUsed(pqIdx);
        assertTrue("DataFusion used after free=" + dfAfter + " should be < before=" + dfUsed, dfAfter < dfUsed);
        assertTrue("Parquet used after free=" + pqAfter + " should be < before=" + pqUsed, pqAfter < pqUsed);

        // Cleanup
        NativeBridge.shutdownTokioRuntimeManager();
    }
}
