#pragma once

#include "vectordb/index/index_interface.hpp"
#include <unordered_map>
#include <vector>
#include <random>
#include <shared_mutex>

namespace vectordb {

struct HNSWParams {
    size_t M{16};                   // Max edges per node at level > 0 (level 0 has 2*M)
    size_t ef_construction{200};     // Search queue size during build
    size_t ef_search{50};           // Search queue size during query
    double mL{1.0 / std::log(16.0)};// Level multiplier
    uint32_t random_seed{100};
};

class HNSWIndex : public VectorIndex {
public:
    using HNSWParams = vectordb::HNSWParams;

    HNSWIndex(size_t dimension, DistanceMetric metric, HNSWParams params = HNSWParams{});

    void add(VectorId id, const Vector& vector) override;
    bool remove(VectorId id) override;
    bool contains(VectorId id) const override;
    std::optional<Vector> get_vector(VectorId id) const override;

    std::vector<SearchResult> search(
        const Vector& query,
        size_t top_k,
        const IdFilterPredicate& filter_fn = nullptr,
        size_t ef_search = 0
    ) const override;

    size_t size() const override;
    size_t dimension() const override { return dimension_; }
    DistanceMetric metric() const override { return metric_; }
    IndexType type() const override { return IndexType::HNSW; }
    void clear() override;

    void serialize(std::ostream& out) const override;
    void deserialize(std::istream& in) override;

    const HNSWParams& params() const { return params_; }

private:
    struct Node {
        VectorId id{0};
        Vector values;
        int level{0};
        // friends[l] contains the internal node indices connected at layer l
        std::vector<std::vector<size_t>> friends;
        bool deleted{false};
    };

    size_t dimension_;
    DistanceMetric metric_;
    HNSWParams params_;

    std::vector<Node> nodes_;
    std::unordered_map<VectorId, size_t> id_to_index_;
    int max_level_{-1};
    size_t enter_point_{0};
    size_t active_count_{0};

    mutable std::default_random_engine rng_;
    mutable std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};

    int random_level();
    float distance(const float* a, const float* b) const;

    // Search layer at specified level
    std::vector<std::pair<float, size_t>> search_layer(
        const float* query_data,
        const std::vector<size_t>& enter_points,
        size_t ef,
        int level
    ) const;

    // Select M neighbors from candidate set
    std::vector<size_t> select_neighbors(
        const float* query_data,
        std::vector<std::pair<float, size_t>>& candidates,
        size_t max_m
    ) const;
};

} // namespace vectordb
