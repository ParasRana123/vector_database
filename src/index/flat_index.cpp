#include "vectordb/index/flat_index.hpp"
#include "vectordb/core/distance.hpp"
#include <queue>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace vectordb {

FlatIndex::FlatIndex(size_t dimension, DistanceMetric metric)
    : dimension_(dimension), metric_(metric) {}

void FlatIndex::add(VectorId id, const Vector& vector) {
    if (vector.size() != dimension_) {
        throw std::invalid_argument("Vector dimension mismatch. Expected: " + 
            std::to_string(dimension_) + ", got: " + std::to_string(vector.size()));
    }

    auto it = id_to_index_.find(id);
    if (it != id_to_index_.end()) {
        // Update existing vector in-place
        size_t idx = it->second;
        std::memcpy(data_.data() + idx * dimension_, vector.data(), dimension_ * sizeof(float));
        return;
    }

    size_t new_idx = ids_.size();
    ids_.push_back(id);
    id_to_index_[id] = new_idx;

    size_t old_size = data_.size();
    data_.resize(old_size + dimension_);
    std::memcpy(data_.data() + old_size, vector.data(), dimension_ * sizeof(float));
}

bool FlatIndex::remove(VectorId id) {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) {
        return false;
    }

    size_t idx_to_remove = it->second;
    size_t last_idx = ids_.size() - 1;

    if (idx_to_remove != last_idx) {
        VectorId last_id = ids_[last_idx];
        // Move last element's vector data to the removed slot
        std::memcpy(data_.data() + idx_to_remove * dimension_,
                    data_.data() + last_idx * dimension_,
                    dimension_ * sizeof(float));
        ids_[idx_to_remove] = last_id;
        id_to_index_[last_id] = idx_to_remove;
    }

    ids_.pop_back();
    data_.resize(ids_.size() * dimension_);
    id_to_index_.erase(it);
    return true;
}

bool FlatIndex::contains(VectorId id) const {
    return id_to_index_.find(id) != id_to_index_.end();
}

std::optional<Vector> FlatIndex::get_vector(VectorId id) const {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) {
        return std::nullopt;
    }
    size_t idx = it->second;
    Vector v(dimension_);
    std::memcpy(v.data(), data_.data() + idx * dimension_, dimension_ * sizeof(float));
    return v;
}

std::vector<SearchResult> FlatIndex::search(
    const Vector& query,
    size_t top_k,
    const IdFilterPredicate& filter_fn,
    size_t /* ef_search */
) const {
    if (query.size() != dimension_) {
        throw std::invalid_argument("Query vector dimension mismatch. Expected: " +
            std::to_string(dimension_) + ", got: " + std::to_string(query.size()));
    }
    if (top_k == 0 || ids_.empty()) {
        return {};
    }

    // Min-heap storing pairs of (score, SearchResult) to keep top-k elements
    struct MinHeapCompare {
        bool operator()(const SearchResult& a, const SearchResult& b) const {
            return a.score > b.score; // min-heap by score
        }
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, MinHeapCompare> min_heap;

    const float* q_ptr = query.data();
    for (size_t i = 0; i < ids_.size(); ++i) {
        VectorId id = ids_[i];
        if (filter_fn && !filter_fn(id)) {
            continue;
        }

        const float* vec_ptr = data_.data() + i * dimension_;
        float score = compute_score(q_ptr, vec_ptr, dimension_, metric_);

        if (min_heap.size() < top_k) {
            min_heap.push(SearchResult(id, score));
        } else if (score > min_heap.top().score) {
            min_heap.pop();
            min_heap.push(SearchResult(id, score));
        }
    }

    std::vector<SearchResult> results;
    results.reserve(min_heap.size());
    while (!min_heap.empty()) {
        results.push_back(min_heap.top());
        min_heap.pop();
    }
    // Reverse so highest score is first
    std::reverse(results.begin(), results.end());
    return results;
}

void FlatIndex::clear() {
    ids_.clear();
    data_.clear();
    id_to_index_.clear();
}

void FlatIndex::serialize(std::ostream& out) const {
    uint32_t magic = 0x56444246; // "VDBF" (Vector DB Flat)
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

    uint32_t dim = static_cast<uint32_t>(dimension_);
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

    uint32_t metric_val = static_cast<uint32_t>(metric_);
    out.write(reinterpret_cast<const char*>(&metric_val), sizeof(metric_val));

    uint64_t count = ids_.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    if (count > 0) {
        out.write(reinterpret_cast<const char*>(ids_.data()), count * sizeof(VectorId));
        out.write(reinterpret_cast<const char*>(data_.data()), count * dimension_ * sizeof(float));
    }
}

void FlatIndex::deserialize(std::istream& in) {
    clear();
    uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x56444246) {
        throw std::runtime_error("Invalid FlatIndex binary header magic");
    }

    uint32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    dimension_ = dim;

    uint32_t metric_val = 0;
    in.read(reinterpret_cast<char*>(&metric_val), sizeof(metric_val));
    metric_ = static_cast<DistanceMetric>(metric_val);

    uint64_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));

    ids_.resize(count);
    data_.resize(count * dimension_);

    if (count > 0) {
        in.read(reinterpret_cast<char*>(ids_.data()), count * sizeof(VectorId));
        in.read(reinterpret_cast<char*>(data_.data()), count * dimension_ * sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            id_to_index_[ids_[i]] = i;
        }
    }
}

} // namespace vectordb
