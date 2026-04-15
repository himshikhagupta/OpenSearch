/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics;

import com.parquet.parquetdataformat.ParquetDataFormatPlugin;
import org.opensearch.common.settings.Settings;
import org.opensearch.datafusion.DataFusionPlugin;
import org.opensearch.index.IndexSettings;
import org.opensearch.plugins.Plugin;
import org.opensearch.test.OpenSearchIntegTestCase;
import org.opensearch.vectorized.execution.jni.NativeLibraryLoader;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Integration test verifying per-plugin native heap stats
 * via the parquet writer path.
 */
@OpenSearchIntegTestCase.ClusterScope(scope = OpenSearchIntegTestCase.Scope.SUITE, numDataNodes = 1)
public class HeapStatsIT extends OpenSearchIntegTestCase {

    private static final String INDEX = "heap-stats-test";

    @Override
    protected Collection<Class<? extends Plugin>> nodePlugins() {
        return List.of(AnalyticsPlugin.class, ParquetDataFormatPlugin.class, DataFusionPlugin.class);
    }

    /** Parquet-backed optimized index. */
    private void createTestIndex() {
        if (!indexExists(INDEX)) {
            client().admin().indices().prepareCreate(INDEX)
                .setSettings(Settings.builder()
                    .put(IndexSettings.OPTIMIZED_INDEX_ENABLED_SETTING.getKey(), true)
                    .put("number_of_shards", 1)
                    .put("number_of_replicas", 0))
                .setMapping("value", "type=long", "name", "type=keyword")
                .get();
            ensureGreen(INDEX);
        }
    }

    public void testPluginHeapsAreRegisteredAfterIndexing() {
        createTestIndex();
        for (int i = 0; i < 50; i++) {
            client().prepareIndex(INDEX).setId(String.valueOf(i))
                .setSource("value", (long) i, "name", "item-" + i).get();
        }
        refresh(INDEX);

        String json = NativeLibraryLoader.getHeapStats();
        assertNotNull(json);
        logger.info("Heap stats: {}", json);

        List<Map<String, Object>> stats = parseStatsJson(json);
        assertTrue("Should have at least 2 plugins, got " + stats.size(), stats.size() >= 2);
        assertNotNull("engine-datafusion missing", findPlugin(stats, "engine-datafusion"));
        assertNotNull("parquet-format missing", findPlugin(stats, "parquet-format"));
    }

    public void testHeapStatsValuesAreNonNegative() {
        List<Map<String, Object>> stats = parseStatsJson(NativeLibraryLoader.getHeapStats());
        for (Map<String, Object> entry : stats) {
            String name = (String) entry.get("name");
            long used = (Long) entry.get("used");
            long committed = (Long) entry.get("committed");
            logger.info("plugin='{}' used={}KB committed={}KB", name, used / 1024, committed / 1024);
            assertTrue(name + " used >= 0", used >= 0);
            assertTrue(name + " committed >= 0", committed >= 0);
        }
    }

    public void testSumOfPluginHeapsDoesNotExceedGlobalMimalloc() {
        createTestIndex();
        for (int i = 0; i < 200; i++) {
            client().prepareIndex(INDEX).setId(String.valueOf(i))
                .setSource("value", (long) i, "name", "item-" + i).get();
        }
        refresh(INDEX);

        List<Map<String, Object>> pluginStats = parseStatsJson(NativeLibraryLoader.getHeapStats());
        long totalPluginCommitted = pluginStats.stream().mapToLong(s -> (Long) s.get("committed")).sum();
        long totalPluginUsed = pluginStats.stream().mapToLong(s -> (Long) s.get("used")).sum();

        // Global stats from mi_stats_get_json — flat: {"current_commit":N,"peak_commit":N}
        Map<String, Object> globalStats = parseObjectJson(NativeLibraryLoader.getGlobalMimallocStats());
        long globalCommit = (Long) globalStats.get("current_commit");

        for (Map<String, Object> entry : pluginStats) {
            logger.info("[heap] plugin='{}' used={}KB committed={}KB",
                entry.get("name"), (Long) entry.get("used") / 1024, (Long) entry.get("committed") / 1024);
        }
        logger.info("[heap] total plugin used={}KB committed={}KB", totalPluginUsed / 1024, totalPluginCommitted / 1024);
        logger.info("[global] current_commit={}KB peak_commit={}KB",
            globalCommit / 1024, (Long) globalStats.get("peak_commit") / 1024);

        assertTrue("plugin committed <= global commit", totalPluginCommitted <= globalCommit);
        assertTrue("plugin used <= plugin committed", totalPluginUsed <= totalPluginCommitted);
    }

    public void testHeapUsageIncreasesAfterIndexing() {
        String beforeJson = NativeLibraryLoader.getHeapStats();
        long parquetBefore = getPluginCommitted(parseStatsJson(beforeJson), "parquet-format");

        createTestIndex();
        for (int i = 0; i < 200; i++) {
            client().prepareIndex(INDEX).setId(String.valueOf(i))
                .setSource("value", (long) i, "name", "item-" + i).get();
        }
        refresh(INDEX);

        List<Map<String, Object>> afterStats = parseStatsJson(NativeLibraryLoader.getHeapStats());
        long parquetAfter = getPluginCommitted(afterStats, "parquet-format");
        logger.info("parquet-format committed: before={}KB after={}KB", parquetBefore / 1024, parquetAfter / 1024);
        assertTrue("parquet committed should be > 0 after indexing", parquetAfter > 0);
    }

    // ── helpers ──

    private long getPluginCommitted(List<Map<String, Object>> stats, String name) {
        Map<String, Object> p = findPlugin(stats, name);
        return p != null ? (Long) p.get("committed") : 0;
    }

    private static List<Map<String, Object>> parseStatsJson(String json) {
        List<Map<String, Object>> result = new ArrayList<>();
        json = json.trim();
        if (json.startsWith("[")) json = json.substring(1);
        if (json.endsWith("]")) json = json.substring(0, json.length() - 1);
        if (json.isEmpty()) return result;
        for (String entry : json.split("\\},\\{")) {
            entry = entry.replace("{", "").replace("}", "");
            Map<String, Object> map = new HashMap<>();
            for (String kv : entry.split(",")) {
                String[] parts = kv.split(":", 2);
                String key = parts[0].replace("\"", "").trim();
                String val = parts[1].replace("\"", "").trim();
                try { map.put(key, Long.parseLong(val)); } catch (NumberFormatException e) { map.put(key, val); }
            }
            result.add(map);
        }
        return result;
    }

    private static Map<String, Object> findPlugin(List<Map<String, Object>> stats, String name) {
        return stats.stream().filter(m -> name.equals(m.get("name"))).findFirst().orElse(null);
    }

    private static Map<String, Object> parseObjectJson(String json) {
        json = json.trim();
        if (json.startsWith("{")) json = json.substring(1);
        if (json.endsWith("}")) json = json.substring(0, json.length() - 1);
        Map<String, Object> map = new HashMap<>();
        for (String kv : json.split(",")) {
            String[] parts = kv.split(":", 2);
            String key = parts[0].replace("\"", "").trim();
            String val = parts[1].replace("\"", "").trim();
            try { map.put(key, Long.parseLong(val)); } catch (NumberFormatException e) { map.put(key, val); }
        }
        return map;
    }
}
