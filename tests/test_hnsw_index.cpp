#include "vectordb/index/hnsw_index.hpp"
#include "vectordb/index/flat_index.hpp"
#include "vectordb/core/distance.hpp"
#include <iostream>
#include <cassert>
#include <random>
#include <sstream>
#include <unordered_set>

using namespace vectordb;

void test_hnsw_basic() {
    HNSWIndex index(4, DistanceMetric::EUCLIDEAN);
    index.add(1, {1.0f, 0.0f, 0.0f, 0.0f});
    index.add(2, {0.0f, 1.0f, 0.0f, 0.0f});
    index.add(3, {0.0f, 0.0f, 1.0f, 0.0f});

    assert(index.size() == 3);
    assert(index.contains(1));
    assert(index.contains(2));
    assert(!index.contains(99));

    auto res = index.search({0.9f, 0.1f, 0.0f, 0.0f}, 1);
    assert(!res.empty());
    assert(res[0].id == 1);

    bool rem = index.remove(2);
    assert(rem);
    assert(index.size() == 2);
    assert(!index.contains(2));

    std::cout << "[PASS] test_hnsw_basic\n";
}

void test_hnsw_recall_vs_flat() {
    const size_t dim = 32;
    const size_t num_vectors = 500;
    const size_t num_queries = 20;
    const size_t top_k = 5;

    FlatIndex flat(dim, DistanceMetric::COSINE);
    HNSWIndex hnsw(dim, DistanceMetric::COSINE);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 1; i <= num_vectors; ++i) {
        Vector v(dim);
        for (size_t d = 0; d < dim; ++d) v[d] = dist(rng);
        l2_normalize(v);

        flat.add(i, v);
        hnsw.add(i, v);
    }

    size_t total_hits = 0;
    size_t total_possible = num_queries * top_k;

    for (size_t q = 0; q < num_queries; ++q) {
        Vector query(dim);
        for (size_t d = 0; d < dim; ++d) query[d] = dist(rng);
        l2_normalize(query);

        auto ground_truth = flat.search(query, top_k);
        auto hnsw_results = hnsw.search(query, top_k);

        std::unordered_set<VectorId> gt_ids;
        for (const auto& r : ground_truth) gt_ids.insert(r.id);

        for (const auto& r : hnsw_results) {
            if (gt_ids.find(r.id) != gt_ids.end()) {
                ++total_hits;
            }
        }
    }

    double recall = static_cast<double>(total_hits) / static_cast<double>(total_possible);
    std::cout << "HNSW Recall@" << top_k << ": " << (recall * 100.0) << "%\n";
    assert(recall >= 0.90); // Expect >= 90% recall
    std::cout << "[PASS] test_hnsw_recall_vs_flat\n";
}

void test_hnsw_serialization() {
    HNSWIndex original(3, DistanceMetric::COSINE);
    original.add(10, {1.0f, 0.0f, 0.0f});
    original.add(20, {0.0f, 1.0f, 0.0f});
    original.add(30, {0.0f, 0.0f, 1.0f});

    std::stringstream ss;
    original.serialize(ss);

    HNSWIndex loaded(1, DistanceMetric::EUCLIDEAN);
    loaded.deserialize(ss);

    assert(loaded.size() == 3);
    assert(loaded.dimension() == 3);
    assert(loaded.contains(10));
    assert(loaded.contains(20));
    assert(loaded.contains(30));

    auto res = loaded.search({0.8f, 0.1f, 0.0f}, 1);
    assert(!res.empty());
    assert(res[0].id == 10);

    std::cout << "[PASS] test_hnsw_serialization\n";
}

int main() {
    std::cout << "--- Running HNSW Index Tests ---\n";
    test_hnsw_basic();
    test_hnsw_recall_vs_flat();
    test_hnsw_serialization();
    std::cout << "All HNSW Index tests passed successfully!\n";
    return 0;
}
