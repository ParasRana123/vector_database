#pragma once

#include "vectordb/core/types.hpp"
#include "vectordb/core/metadata.hpp"
#include "vectordb/core/thread_pool.hpp"
#include "vectordb/engine/collection.hpp"
#include "vectordb/embedding/embedder.hpp"
#include "vectordb/storage/wal.hpp"
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace vectordb {

struct DatabaseConfig {
    std::string data_dir{"./vectordb_data"};
    std::string model_dir{"./models/all-MiniLM-L6-v2"};
    bool enable_wal{true};
    size_t num_threads{0};
};

struct DatabaseStats {
    size_t total_collections{0};
    size_t total_documents{0};
    std::vector<std::string> collection_names;
    bool embedder_ready{false};
};

class Database {
public:
    explicit Database(DatabaseConfig config = DatabaseConfig{});
    ~Database();

    // Collection Management
    bool create_collection(CollectionConfig config);
    std::shared_ptr<Collection> get_collection(const std::string& name);
    bool drop_collection(const std::string& name);
    bool has_collection(const std::string& name) const;
    std::vector<std::string> list_collections() const;

    // Direct Vector Operations
    void insert_vector(
        const std::string& collection_name,
        VectorId id,
        const Vector& vector,
        const std::string& payload = "",
        const Metadata& metadata = {}
    );

    std::vector<SearchResult> search_vector(
        const std::string& collection_name,
        const Vector& query,
        size_t top_k = 10,
        const std::shared_ptr<Filter>& filter = nullptr,
        size_t ef_search = 0
    );

    // High-Level Text / Semantic Search Operations
    VectorId insert_text(
        const std::string& collection_name,
        VectorId id,
        const std::string& text,
        const Metadata& metadata = {}
    );

    std::vector<SearchResult> search_text(
        const std::string& collection_name,
        const std::string& query_text,
        size_t top_k = 10,
        const std::shared_ptr<Filter>& filter = nullptr,
        size_t ef_search = 0
    );

    // Persistence & Snapshotting
    bool save_all(const std::string& checkpoint_dir = "");
    bool load_all(const std::string& checkpoint_dir = "");

    // Text Embedder access
    TextEmbedder& embedder() { return embedder_; }
    ThreadPool& thread_pool() { return *thread_pool_; }
    DatabaseStats get_stats() const;

private:
    DatabaseConfig config_;
    std::unique_ptr<ThreadPool> thread_pool_;
    TextEmbedder embedder_;

    mutable std::shared_mutex db_mutex_;
    std::unordered_map<std::string, std::shared_ptr<Collection>> collections_;
    std::unordered_map<std::string, std::unique_ptr<WAL>> wals_;

    void init_data_dir();
    void recover_wal(const std::string& coll_name, Collection& coll);
};

} // namespace vectordb
