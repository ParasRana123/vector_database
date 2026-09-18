#pragma once

#include "vectordb/index/index_interface.hpp"
#include <unordered_map>
#include <vector>

namespace vectordb {

class FlatIndex : public VectorIndex {
public:
    FlatIndex(size_t dimension, DistanceMetric metric);

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

    size_t size() const override { return ids_.size(); }
    size_t dimension() const override { return dimension_; }
    DistanceMetric metric() const override { return metric_; }
    IndexType type() const override { return IndexType::FLAT; }
    void clear() override;

    void serialize(std::ostream& out) const override;
    void deserialize(std::istream& in) override;

private:
    size_t dimension_;
    DistanceMetric metric_;
    std::vector<VectorId> ids_;
    std::vector<float> data_; // Flattened buffer of size (N * dimension_)
    std::unordered_map<VectorId, size_t> id_to_index_;
};

} // namespace vectordb
