#include "vectordb/index/flat_index.hpp"
#include "vectordb/index/hnsw_index.hpp"
#include "vectordb/core/thread_pool.hpp"
#include "vectordb/core/distance.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <algorithm>

using namespace vectordb;

void run_benchmark(size_t num_vectors, size_t dim, size_t num_queries, size_t top_k) {
    std::cout << "\n=======================================================\n";
    std::cout << "                 VECTORDB V1 BENCHMARK                 \n";
    std::cout << "=======================================================\n";
    std::cout << "Dataset: " << num_vectors << " vectors | Dim: " << dim 
              << " | Metric: COSINE | Queries: " << num_queries << " | Top-K: " << top_k << "\n\n";

    std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::cout << "Generating synthetic data...";
    std::vector<Vector> data(num_vectors, Vector(dim));
    for (size_t i = 0; i < num_vectors; ++i) {
        for (size_t d = 0; d < dim; ++d) data[i][d] = dist(rng);
        l2_normalize(data[i]);
    }
    std::vector<Vector> queries(num_queries, Vector(dim));
    for (size_t i = 0; i < num_queries; ++i) {
        for (size_t d = 0; d < dim; ++d) queries[i][d] = dist(rng);
        l2_normalize(queries[i]);
    }
    std::cout << " Done.\n";

    // 1. Flat Index Build & Search
    FlatIndex flat(dim, DistanceMetric::COSINE);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_vectors; ++i) {
        flat.add(i + 1, data[i]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double flat_build_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

    std::cout << "\n[Flat Index] Build Time: " << flat_build_ms << " ms (" 
              << static_cast<size_t>(num_vectors / (flat_build_ms / 1000.0)) << " vec/s)\n";

    std::vector<std::vector<SearchResult>> ground_truth(num_queries);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < num_queries; ++q) {
        ground_truth[q] = flat.search(queries[q], top_k);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double flat_search_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "[Flat Index] Avg Search Latency: " << (flat_search_us / num_queries) << " us ("
              << static_cast<size_t>(num_queries / (flat_search_us / 1000000.0)) << " QPS)\n";

    // 2. HNSW Index Build & Search
    HNSWIndex::HNSWParams params;
    params.M = 16;
    params.ef_construction = 100;
    params.ef_search = 50;

    HNSWIndex hnsw(dim, DistanceMetric::COSINE, params);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_vectors; ++i) {
        hnsw.add(i + 1, data[i]);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double hnsw_build_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

    std::cout << "\n[HNSW Index] Build Time: " << hnsw_build_ms << " ms (" 
              << static_cast<size_t>(num_vectors / (hnsw_build_ms / 1000.0)) << " vec/s)\n";

    std::vector<double> latencies_us;
    latencies_us.reserve(num_queries);
    size_t total_hits = 0;

    for (size_t q = 0; q < num_queries; ++q) {
        auto q_start = std::chrono::high_resolution_clock::now();
        auto res = hnsw.search(queries[q], top_k);
        auto q_end = std::chrono::high_resolution_clock::now();
        latencies_us.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(q_end - q_start).count() / 1000.0);

        std::unordered_set<VectorId> gt_set;
        for (const auto& r : ground_truth[q]) gt_set.insert(r.id);
        for (const auto& r : res) {
            if (gt_set.find(r.id) != gt_set.end()) ++total_hits;
        }
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    double avg_lat = 0.0;
    for (double lat : latencies_us) avg_lat += lat;
    avg_lat /= num_queries;

    double p50 = latencies_us[num_queries / 2];
    double p95 = latencies_us[static_cast<size_t>(num_queries * 0.95)];
    double p99 = latencies_us[static_cast<size_t>(num_queries * 0.99)];
    double recall = static_cast<double>(total_hits) / (num_queries * top_k);

    std::cout << "[HNSW Index] Search Performance:\n";
    std::cout << "  - Recall@" << top_k << ": " << std::fixed << std::setprecision(2) << (recall * 100.0) << "%\n";
    std::cout << "  - Average Latency: " << std::fixed << std::setprecision(1) << avg_lat << " us\n";
    std::cout << "  - P50 Latency:     " << p50 << " us\n";
    std::cout << "  - P95 Latency:     " << p95 << " us\n";
    std::cout << "  - P99 Latency:     " << p99 << " us\n";
    std::cout << "  - Throughput:      " << static_cast<size_t>(1000000.0 / avg_lat) << " QPS (single-thread)\n";
    std::cout << "=======================================================\n\n";
}

int main(int argc, char* argv[]) {
    size_t num_vectors = 2000;
    size_t dim = 128;
    size_t num_queries = 100;
    size_t top_k = 10;

    if (argc > 1) num_vectors = std::stoull(argv[1]);
    if (argc > 2) dim = std::stoull(argv[2]);

    run_benchmark(num_vectors, dim, num_queries, top_k);
    return 0;
}
