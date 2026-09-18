#pragma once

#include "vectordb/core/types.hpp"
#include <vector>
#include <optional>
#include <functional>
#include <iostream>
#include <memory>

namespace vectordb {

using IdFilterPredicate = std::function<bool(VectorId)>;

class VectorIndex {
public:
    virtual ~VectorIndex() = default;

    // Insertion
    virtual void add(VectorId id, const Vector& vector) = 0;
    virtual void add_batch(const std::vector<VectorId>& ids, const std::vector<Vector>& vectors) {
        for (size_t i = 0; i < ids.size(); ++i) {
            add(ids[i], vectors[i]);
        }
    }

    // Modification & Deletion
    virtual bool remove(VectorId id) = 0;
    virtual bool contains(VectorId id) const = 0;
    virtual std::optional<Vector> get_vector(VectorId id) const = 0;

    // Search
    virtual std::vector<SearchResult> search(
        const Vector& query,
        size_t top_k,
        const IdFilterPredicate& filter_fn = nullptr,
        size_t ef_search = 0
    ) const = 0;

    // Metadata & Size
    virtual size_t size() const = 0;
    virtual size_t dimension() const = 0;
    virtual DistanceMetric metric() const = 0;
    virtual IndexType type() const = 0;
    virtual void clear() = 0;

    // Persistence
    virtual void serialize(std::ostream& out) const = 0;
    virtual void deserialize(std::istream& in) = 0;
};

} // namespace vectordb
