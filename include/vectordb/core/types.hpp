#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <sstream>

namespace vectordb {

using VectorId = uint64_t;
using Vector = std::vector<float>;

enum class DistanceMetric {
    COSINE,
    EUCLIDEAN,
    DOT_PRODUCT
};

inline std::string distance_metric_to_string(DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::COSINE: return "COSINE";
        case DistanceMetric::EUCLIDEAN: return "EUCLIDEAN";
        case DistanceMetric::DOT_PRODUCT: return "DOT_PRODUCT";
        default: return "UNKNOWN";
    }
}

inline DistanceMetric parse_distance_metric(const std::string& str) {
    std::string upper = str;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if (upper == "COSINE" || upper == "COS") return DistanceMetric::COSINE;
    if (upper == "EUCLIDEAN" || upper == "L2") return DistanceMetric::EUCLIDEAN;
    if (upper == "DOT_PRODUCT" || upper == "DOT" || upper == "IP") return DistanceMetric::DOT_PRODUCT;
    throw std::invalid_argument("Unknown distance metric: " + str);
}

enum class IndexType {
    FLAT,
    HNSW
};

inline std::string index_type_to_string(IndexType type) {
    switch (type) {
        case IndexType::FLAT: return "FLAT";
        case IndexType::HNSW: return "HNSW";
        default: return "UNKNOWN";
    }
}

inline IndexType parse_index_type(const std::string& str) {
    std::string upper = str;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if (upper == "FLAT") return IndexType::FLAT;
    if (upper == "HNSW") return IndexType::HNSW;
    throw std::invalid_argument("Unknown index type: " + str);
}

struct SearchResult {
    VectorId id{0};
    float score{0.0f};
    std::string payload{""};

    SearchResult() = default;
    SearchResult(VectorId id_, float score_, std::string payload_ = "")
        : id(id_), score(score_), payload(std::move(payload_)) {}

    bool operator<(const SearchResult& other) const {
        return score < other.score;
    }

    bool operator>(const SearchResult& other) const {
        return score > other.score;
    }
};

struct VectorRecord {
    VectorId id{0};
    Vector values;
    std::string payload;

    VectorRecord() = default;
    VectorRecord(VectorId id_, Vector values_, std::string payload_ = "")
        : id(id_), values(std::move(values_)), payload(std::move(payload_)) {}
};

struct CollectionConfig {
    std::string name;
    size_t dimension{384};
    DistanceMetric metric{DistanceMetric::COSINE};
    IndexType index_type{IndexType::HNSW};
    size_t hnsw_m{16};
    size_t hnsw_ef_construction{200};
    size_t hnsw_ef_search{50};
};

struct QueryOptions {
    size_t top_k{10};
    size_t ef_search{50};
    float score_threshold{0.0f};
    bool has_score_threshold{false};
};

} // namespace vectordb
