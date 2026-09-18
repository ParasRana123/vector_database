#pragma once

#include "vectordb/core/types.hpp"
#include "vectordb/core/metadata.hpp"
#include "vectordb/index/index_interface.hpp"
#include <string>
#include <memory>
#include <unordered_map>
#include <shared_mutex>
#include <optional>

namespace vectordb {

struct DocumentRecord {
    VectorId id{0};
    Vector vector;
    std::string payload;
    Metadata metadata;
};

class Collection {
public:
    explicit Collection(CollectionConfig config);
    ~Collection() = default;

    Collection(const Collection&) = delete;
    Collection& operator=(const Collection&) = delete;

    const CollectionConfig& config() const { return config_; }
    const std::string& name() const { return config_.name; }
    size_t dimension() const { return config_.dimension; }
    DistanceMetric metric() const { return config_.metric; }
    IndexType index_type() const { return config_.index_type; }

    void insert(
        VectorId id,
        const Vector& vector,
        std::string payload = "",
        Metadata metadata = {}
    );

    void insert_batch(
        const std::vector<VectorId>& ids,
        const std::vector<Vector>& vectors,
        const std::vector<std::string>& payloads = {},
        const std::vector<Metadata>& metadatas = {}
    );

    bool remove(VectorId id);
    bool contains(VectorId id) const;
    std::optional<DocumentRecord> get(VectorId id) const;

    std::vector<SearchResult> search(
        const Vector& query,
        size_t top_k,
        const std::shared_ptr<Filter>& filter = nullptr,
        size_t ef_search = 0
    ) const;

    size_t size() const;
    void clear();

    bool save_snapshot(const std::string& filepath) const;
    bool load_snapshot(const std::string& filepath);

private:
    CollectionConfig config_;
    std::unique_ptr<VectorIndex> index_;
    std::unordered_map<VectorId, std::string> payloads_;
    std::unordered_map<VectorId, Metadata> metadata_;
    mutable std::shared_mutex mutex_;

    void init_index();
};

} // namespace vectordb
