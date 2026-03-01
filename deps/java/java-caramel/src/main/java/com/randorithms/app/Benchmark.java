package com.randorithms.app;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import it.unimi.dsi.bits.TransformationStrategies;
import it.unimi.dsi.fastutil.longs.LongArrayList;
import it.unimi.dsi.sux4j.mph.GV3CompressedFunction;
import it.unimi.dsi.sux4j.mph.codec.Codec;

public class Benchmark {

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: Benchmark <keys_file> <values_file> <seed> <num_queries>");
            System.exit(1);
        }

        String keysFile = args[0];
        String valuesFile = args[1];
        int seed = Integer.parseInt(args[2]);
        int numQueries = Integer.parseInt(args[3]);

        List<String> keys = readKeys(keysFile);
        long[] values = readValues(valuesFile, keys.size());

        LongArrayList valuesList = new LongArrayList(values);

        long t0 = System.nanoTime();
        GV3CompressedFunction<CharSequence> csf = new GV3CompressedFunction.Builder<CharSequence>()
                .keys(keys)
                .transform(TransformationStrategies.rawUtf16())
                .values(valuesList)
                .codec(new Codec.Huffman())
                .build();
        long t1 = System.nanoTime();
        double constructionTimeS = (t1 - t0) / 1e9;

        long numBits = csf.numBits();

        File tmpFile = File.createTempFile("csf_benchmark_", ".ser");
        try {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(tmpFile))) {
                oos.writeObject(csf);
            }
            long serializedBytes = tmpFile.length();

            long[] inferenceStats = measureInference(csf, keys, seed, numQueries);

            System.out.println("{");
            System.out.println("  \"method\": \"java_csf\",");
            System.out.printf("  \"construction_time_s\": %.6f,%n", constructionTimeS);
            System.out.println("  \"inference_ns\": {");
            System.out.printf("    \"mean\": %.1f,%n", (double) inferenceStats[0]);
            System.out.printf("    \"median\": %.1f,%n", (double) inferenceStats[1]);
            System.out.printf("    \"std\": %.1f,%n", (double) inferenceStats[2]);
            System.out.printf("    \"min\": %d,%n", inferenceStats[3]);
            System.out.printf("    \"max\": %d,%n", inferenceStats[4]);
            System.out.printf("    \"p95\": %.1f,%n", (double) inferenceStats[5]);
            System.out.printf("    \"p99\": %.1f%n", (double) inferenceStats[6]);
            System.out.println("  },");
            System.out.println("  \"memory\": {");
            System.out.printf("    \"num_bits\": %d,%n", numBits);
            System.out.printf("    \"serialized_bytes\": %d%n", serializedBytes);
            System.out.println("  }");
            System.out.println("}");
        } finally {
            tmpFile.delete();
        }
    }

    private static List<String> readKeys(String path) throws IOException {
        List<String> keys = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(path), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                keys.add(line);
            }
        }
        return keys;
    }

    private static long[] readValues(String path, int expectedCount) throws IOException {
        long[] values = new long[expectedCount];
        try (DataInputStream dis = new DataInputStream(new FileInputStream(path))) {
            for (int i = 0; i < expectedCount; i++) {
                values[i] = dis.readLong();
            }
        }
        return values;
    }

    private static long[] measureInference(GV3CompressedFunction<CharSequence> csf,
                                            List<String> keys, int seed, int numQueries) {
        Random rng = new Random(seed);
        String[] sampleKeys = new String[numQueries];
        for (int i = 0; i < numQueries; i++) {
            sampleKeys[i] = keys.get(rng.nextInt(keys.size()));
        }

        // warmup
        for (int i = 0; i < Math.min(10, numQueries); i++) {
            csf.getLong(sampleKeys[i]);
        }

        long[] times = new long[numQueries];
        for (int i = 0; i < numQueries; i++) {
            long start = System.nanoTime();
            csf.getLong(sampleKeys[i]);
            long end = System.nanoTime();
            times[i] = end - start;
        }

        return computeStats(times);
    }

    private static long[] computeStats(long[] times) {
        long[] sorted = times.clone();
        Arrays.sort(sorted);
        int n = sorted.length;

        long sum = 0;
        for (long t : sorted) sum += t;
        double mean = (double) sum / n;

        double median;
        if (n % 2 == 0) {
            median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
        } else {
            median = sorted[n / 2];
        }

        double sumSqDiff = 0;
        for (long t : sorted) {
            double diff = t - mean;
            sumSqDiff += diff * diff;
        }
        double std = Math.sqrt(sumSqDiff / n);

        long min = sorted[0];
        long max = sorted[n - 1];
        double p95 = percentile(sorted, 95);
        double p99 = percentile(sorted, 99);

        return new long[]{
            Math.round(mean), Math.round(median), Math.round(std),
            min, max, Math.round(p95), Math.round(p99)
        };
    }

    private static double percentile(long[] sorted, int p) {
        double rank = p / 100.0 * (sorted.length - 1);
        int lower = (int) Math.floor(rank);
        int upper = (int) Math.ceil(rank);
        if (lower == upper) return sorted[lower];
        double frac = rank - lower;
        return sorted[lower] * (1 - frac) + sorted[upper] * frac;
    }
}
