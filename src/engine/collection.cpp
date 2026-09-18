#include "vectordb/engine/collection.hpp"
#include "vectordb/index/flat_index.hpp"
#include "vectordb/index/hnsw_index.hpp"
#include "vectordb/storage/serializer.hpp"
#include <mutex>
#include <shared_mutex>
#include <stdexcept>

namespace vectordb {

Collection::Collection(CollectionConfig config) : config_(std::move(config)) {
    init_index();
}

void Collection::init_index() {
    if (config_.index_type == IndexType::FLAT) {
        index_ = std::make_unique<FlatIndex>(config_.dimension, config_.metric);
    } else {
        HNSWIndex::HNSWParams params;
        params.M = config_.hnsw_m;
        params.ef_construction = config_.hnsw_ef_construction;
        params.ef_search = config_.hnsw_ef_search;
        index_ = std::make_unique<HNSWIndex>(config_.dimension, config_.metric, params);
    }
}

void Collection::insert(
    VectorId id,
    const Vector& vector,
    std::string payload,
    Metadata metadata
) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    index_->add(id, vector);
    if (!payload.empty()) {
        payloads_[id] = std::move(payload);
    }
    if (!metadata.data().empty()) {
        metadata_[id] = std::move(metadata);
    }
}

void Collection::insert_batch(
    const std::vector<VectorId>& ids,
    const std::vector<Vector>& vectors,
    const std::vector<std::string>& payloads,
    const std::vector<Metadata>& metadatas
) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (size_t i = 0; i < ids.size(); ++i) {
        index_->add(ids[i], vectors[i]);
        if (i < payloads.size() && !payloads[i].empty()) {
            payloads_[ids[i]] = payloads[i];
        }
        if (i < metadatas.size() && !metadatas[i].data().empty()) {
            metadata_[ids[i]] = metadatas[i];
        }
    }
}

bool Collection::remove(VectorId id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    bool removed = index_->remove(id);
    payloads_.erase(id);
    metadata_.erase(id);
    return removed;
}

bool Collection::contains(VectorId id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return index_->contains(id);
}

std::optional<DocumentRecord> Collection::get(VectorId id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto vec_opt = index_->get_vector(id);
    if (!vec_opt.has_value()) {
        return std::nullopt;
    }

    DocumentRecord doc;
    doc.id = id;
    doc.vector = std::move(*vec_opt);

    auto it_p = payloads_.find(id);
    if (it_p != payloads_.end()) {
        doc.payload = it_p->second;
    }

    auto it_m = metadata_.find(id);
    if (it_m != metadata_.end()) {
        doc.metadata = it_m->second;
    }

    return doc;
}

std::vector<SearchResult> Collection::search(
    const Vector& query,
    size_t top_k,
    const std::shared_ptr<Filter>& filter,
    size_t ef_search
) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    IdFilterPredicate predicate = nullptr;
    if (filter) {
        predicate = [this, &filter](VectorId id) -> bool {
            auto it = metadata_.find(id);
            if (it == metadata_.end()) {
                Metadata empty_meta;
                return filter->matches(empty_meta);
            }
            return filter->matches(it->second);
        };
    }

    auto results = index_->search(query, top_k, predicate, ef_search);

    // Attach payloads to results
    for (auto& r : results) {
        auto it = payloads_.find(r.id);
        if (it != payloads_.end()) {
            r.payload = it->second;
        }
    }

    return results;
}

size_t Collection::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return index_->size();
}

void Collection::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    index_->clear();
    payloads_.clear();
    metadata_.clear();
}

bool Collection::save_snapshot(const std::string& filepath) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    SnapshotData snapshot;
    snapshot.config = config_;
    snapshot.payloads = payloads_;
    snapshot.metadata = metadata_;

    return Serializer::save_to_file(
        filepath,
        snapshot,
        [this](std::ostream& out) {
            index_->serialize(out);
        }
    );
}

bool Collection::load_snapshot(const std::string& filepath) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    SnapshotData snapshot;

    bool ok = Serializer::load_from_file(
        filepath,
        snapshot,
        [this](std::istream& in) {
            index_->deserialize(in);
        }
    );

    if (ok) {
        config_ = snapshot.config;
        payloads_ = std::move(snapshot.payloads);
        metadata_ = std::move(snapshot.metadata);
    }
    return ok;
}

} // namespace vectordb
