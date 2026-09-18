#include "vectordb/engine/database.hpp"
#include <filesystem>
#include <iostream>
#include <fstream>

namespace vectordb {

Database::Database(DatabaseConfig config)
    : config_(std::move(config)),
      thread_pool_(std::make_unique<ThreadPool>(config_.num_threads)) {
    init_data_dir();
    embedder_.load(config_.model_dir);
}

Database::~Database() {
    for (auto& [name, wal] : wals_) {
        if (wal) wal->close();
    }
}

void Database::init_data_dir() {
    if (!config_.data_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(config_.data_dir, ec);
    }
}

bool Database::create_collection(CollectionConfig coll_config) {
    std::unique_lock<std::shared_mutex> lock(db_mutex_);
    if (collections_.find(coll_config.name) != collections_.end()) {
        return false; // already exists
    }

    auto coll = std::make_shared<Collection>(coll_config);
    collections_[coll_config.name] = coll;

    if (config_.enable_wal && !config_.data_dir.empty()) {
        recover_wal(coll_config.name, *coll);
        std::string wal_path = (std::filesystem::path(config_.data_dir) / (coll_config.name + ".wal")).string();
        auto wal = std::make_unique<WAL>(wal_path);
        wal->open();
        wals_[coll_config.name] = std::move(wal);
    }

    return true;
}

void Database::recover_wal(const std::string& coll_name, Collection& coll) {
    std::string wal_path = (std::filesystem::path(config_.data_dir) / (coll_name + ".wal")).string();
    WAL::replay(wal_path, [&coll](const WALRecord& rec) {
        if (rec.type == WALOpType::INSERT) {
            coll.insert(rec.id, rec.vector, rec.payload, rec.metadata);
        } else if (rec.type == WALOpType::DELETE) {
            coll.remove(rec.id);
        } else if (rec.type == WALOpType::CLEAR) {
            coll.clear();
        }
    });
}

std::shared_ptr<Collection> Database::get_collection(const std::string& name) {
    std::shared_lock<std::shared_mutex> lock(db_mutex_);
    auto it = collections_.find(name);
    if (it != collections_.end()) {
        return it->second;
    }
    return nullptr;
}

bool Database::drop_collection(const std::string& name) {
    std::unique_lock<std::shared_mutex> lock(db_mutex_);
    auto it = collections_.find(name);
    if (it == collections_.end()) {
        return false;
    }

    collections_.erase(it);

    auto it_w = wals_.find(name);
    if (it_w != wals_.end()) {
        if (it_w->second) it_w->second->close();
        wals_.erase(it_w);
        std::string wal_path = (std::filesystem::path(config_.data_dir) / (name + ".wal")).string();
        std::error_code ec;
        std::filesystem::remove(wal_path, ec);
    }

    return true;
}

bool Database::has_collection(const std::string& name) const {
    std::shared_lock<std::shared_mutex> lock(db_mutex_);
    return collections_.find(name) != collections_.end();
}

std::vector<std::string> Database::list_collections() const {
    std::shared_lock<std::shared_mutex> lock(db_mutex_);
    std::vector<std::string> names;
    names.reserve(collections_.size());
    for (const auto& [name, _] : collections_) {
        names.push_back(name);
    }
    return names;
}

void Database::insert_vector(
    const std::string& collection_name,
    VectorId id,
    const Vector& vector,
    const std::string& payload,
    const Metadata& metadata
) {
    auto coll = get_collection(collection_name);
    if (!coll) {
        throw std::invalid_argument("Collection not found: " + collection_name);
    }

    coll->insert(id, vector, payload, metadata);

    if (config_.enable_wal) {
        std::shared_lock<std::shared_mutex> lock(db_mutex_);
        auto it = wals_.find(collection_name);
        if (it != wals_.end() && it->second) {
            it->second->append_insert(id, vector, payload, metadata);
        }
    }
}

std::vector<SearchResult> Database::search_vector(
    const std::string& collection_name,
    const Vector& query,
    size_t top_k,
    const std::shared_ptr<Filter>& filter,
    size_t ef_search
) {
    auto coll = get_collection(collection_name);
    if (!coll) {
        throw std::invalid_argument("Collection not found: " + collection_name);
    }
    return coll->search(query, top_k, filter, ef_search);
}

VectorId Database::insert_text(
    const std::string& collection_name,
    VectorId id,
    const std::string& text,
    const Metadata& metadata
) {
    Vector vec = embedder_.embed(text);
    insert_vector(collection_name, id, vec, text, metadata);
    return id;
}

std::vector<SearchResult> Database::search_text(
    const std::string& collection_name,
    const std::string& query_text,
    size_t top_k,
    const std::shared_ptr<Filter>& filter,
    size_t ef_search
) {
    Vector query_vec = embedder_.embed(query_text);
    return search_vector(collection_name, query_vec, top_k, filter, ef_search);
}

bool Database::save_all(const std::string& checkpoint_dir) {
    std::string target_dir = checkpoint_dir.empty() ? config_.data_dir : checkpoint_dir;
    if (target_dir.empty()) return false;

    std::shared_lock<std::shared_mutex> lock(db_mutex_);
    for (const auto& [name, coll] : collections_) {
        std::string snap_path = (std::filesystem::path(target_dir) / (name + ".vdb")).string();
        if (!coll->save_snapshot(snap_path)) {
            return false;
        }
        auto it_w = wals_.find(name);
        if (it_w != wals_.end() && it_w->second) {
            it_w->second->clear(); // WAL can be safely reset after full snapshot
        }
    }
    return true;
}

bool Database::load_all(const std::string& checkpoint_dir) {
    std::string target_dir = checkpoint_dir.empty() ? config_.data_dir : checkpoint_dir;
    if (target_dir.empty() || !std::filesystem::exists(target_dir)) return false;

    for (const auto& entry : std::filesystem::directory_iterator(target_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".vdb") {
            std::string coll_name = entry.path().stem().string();
            CollectionConfig dummy_cfg;
            dummy_cfg.name = coll_name;
            dummy_cfg.dimension = 384;
            auto coll = std::make_shared<Collection>(dummy_cfg);
            if (coll->load_snapshot(entry.path().string())) {
                std::unique_lock<std::shared_mutex> lock(db_mutex_);
                collections_[coll->name()] = coll;
            }
        }
    }
    return true;
}

DatabaseStats Database::get_stats() const {
    std::shared_lock<std::shared_mutex> lock(db_mutex_);
    DatabaseStats stats;
    stats.total_collections = collections_.size();
    stats.embedder_ready = embedder_.is_loaded();

    for (const auto& [name, coll] : collections_) {
        stats.collection_names.push_back(name);
        stats.total_documents += coll->size();
    }
    return stats;
}

} // namespace vectordb
