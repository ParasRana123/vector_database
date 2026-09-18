#include "vectordb/index/hnsw_index.hpp"
#include "vectordb/core/distance.hpp"
#include <queue>
#include <algorithm>
#include <unordered_set>
#include <cmath>
#include <stdexcept>
#include <cstring>

namespace vectordb {

HNSWIndex::HNSWIndex(size_t dimension, DistanceMetric metric, HNSWParams params)
    : dimension_(dimension), metric_(metric), params_(params), rng_(params.random_seed) {
    if (params_.M == 0) params_.M = 16;
    if (params_.ef_construction == 0) params_.ef_construction = 200;
    if (params_.ef_search == 0) params_.ef_search = 50;
    params_.mL = 1.0 / std::log(static_cast<double>(params_.M));
}

int HNSWIndex::random_level() {
    double r = uniform_dist_(rng_);
    if (r == 0.0) r = 0.0000001;
    double val = -std::log(r) * params_.mL;
    return static_cast<int>(val);
}

float HNSWIndex::distance(const float* a, const float* b) const {
    return compute_distance(a, b, dimension_, metric_);
}

size_t HNSWIndex::size() const {
    return active_count_;
}

bool HNSWIndex::contains(VectorId id) const {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) return false;
    return !nodes_[it->second].deleted;
}

std::optional<Vector> HNSWIndex::get_vector(VectorId id) const {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end() || nodes_[it->second].deleted) {
        return std::nullopt;
    }
    return nodes_[it->second].values;
}

std::vector<std::pair<float, size_t>> HNSWIndex::search_layer(
    const float* query_data,
    const std::vector<size_t>& enter_points,
    size_t ef,
    int level
) const {
    std::unordered_set<size_t> visited;
    // candidates: min-heap (lowest distance first)
    auto cmp_min = [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
        return a.first > b.first;
    };
    std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, decltype(cmp_min)> candidates(cmp_min);

    // nearest_neighbors (W): max-heap (furthest distance first)
    auto cmp_max = [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
        return a.first < b.first;
    };
    std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, decltype(cmp_max)> w(cmp_max);

    for (size_t ep : enter_points) {
        if (ep >= nodes_.size() || nodes_[ep].deleted) continue;
        float d = distance(query_data, nodes_[ep].values.data());
        visited.insert(ep);
        candidates.push({d, ep});
        w.push({d, ep});
    }

    while (!candidates.empty()) {
        auto curr = candidates.top();
        candidates.pop();

        float lower_bound = w.top().first;
        if (curr.first > lower_bound && w.size() >= ef) {
            break;
        }

        const auto& node = nodes_[curr.second];
        if (level < static_cast<int>(node.friends.size())) {
            const auto& neighbors = node.friends[level];
            for (size_t neighbor_idx : neighbors) {
                if (neighbor_idx >= nodes_.size() || visited.find(neighbor_idx) != visited.end()) {
                    continue;
                }
                visited.insert(neighbor_idx);

                if (nodes_[neighbor_idx].deleted) continue;

                float d = distance(query_data, nodes_[neighbor_idx].values.data());
                if (w.size() < ef || d < w.top().first) {
                    candidates.push({d, neighbor_idx});
                    w.push({d, neighbor_idx});
                    if (w.size() > ef) {
                        w.pop();
                    }
                }
            }
        }
    }

    std::vector<std::pair<float, size_t>> result;
    result.reserve(w.size());
    while (!w.empty()) {
        result.push_back(w.top());
        w.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::vector<size_t> HNSWIndex::select_neighbors(
    const float* /* query_data */,
    std::vector<std::pair<float, size_t>>& candidates,
    size_t max_m
) const {
    // Sort candidates by distance ascending
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    std::vector<size_t> result;
    result.reserve(std::min(candidates.size(), max_m));
    for (const auto& cand : candidates) {
        if (result.size() >= max_m) break;
        result.push_back(cand.second);
    }
    return result;
}

void HNSWIndex::add(VectorId id, const Vector& vector) {
    if (vector.size() != dimension_) {
        throw std::invalid_argument("Vector dimension mismatch. Expected: " +
            std::to_string(dimension_) + ", got: " + std::to_string(vector.size()));
    }

    // Check if updating existing
    auto it = id_to_index_.find(id);
    if (it != id_to_index_.end()) {
        remove(id);
    }

    int node_level = random_level();
    size_t node_idx = nodes_.size();

    Node new_node;
    new_node.id = id;
    new_node.values = vector;
    new_node.level = node_level;
    new_node.friends.resize(node_level + 1);
    new_node.deleted = false;

    nodes_.push_back(std::move(new_node));
    id_to_index_[id] = node_idx;
    ++active_count_;

    if (max_level_ < 0) {
        // First element in graph
        max_level_ = node_level;
        enter_point_ = node_idx;
        return;
    }

    size_t curr_obj = enter_point_;
    float curr_dist = distance(vector.data(), nodes_[curr_obj].values.data());

    // 1. Greedy search down from max_level to node_level + 1
    for (int l = max_level_; l > node_level; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            if (l < static_cast<int>(nodes_[curr_obj].friends.size())) {
                for (size_t neighbor_idx : nodes_[curr_obj].friends[l]) {
                    if (neighbor_idx >= nodes_.size() || nodes_[neighbor_idx].deleted) continue;
                    float d = distance(vector.data(), nodes_[neighbor_idx].values.data());
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_obj = neighbor_idx;
                        changed = true;
                    }
                }
            }
        }
    }

    // 2. Search and connect for layers min(max_level, node_level) down to 0
    std::vector<size_t> enter_points = {curr_obj};
    for (int l = std::min(max_level_, node_level); l >= 0; --l) {
        auto candidates = search_layer(vector.data(), enter_points, params_.ef_construction, l);
        size_t max_m = (l == 0) ? (2 * params_.M) : params_.M;
        auto neighbors = select_neighbors(vector.data(), candidates, max_m);

        nodes_[node_idx].friends[l] = neighbors;

        // Add back-links
        for (size_t n_idx : neighbors) {
            nodes_[n_idx].friends[l].push_back(node_idx);

            // Prune neighbor's links if over max_m
            if (nodes_[n_idx].friends[l].size() > max_m) {
                std::vector<std::pair<float, size_t>> n_cands;
                for (size_t cand_idx : nodes_[n_idx].friends[l]) {
                    if (!nodes_[cand_idx].deleted) {
                        float d = distance(nodes_[n_idx].values.data(), nodes_[cand_idx].values.data());
                        n_cands.push_back({d, cand_idx});
                    }
                }
                nodes_[n_idx].friends[l] = select_neighbors(nodes_[n_idx].values.data(), n_cands, max_m);
            }
        }

        enter_points.clear();
        for (const auto& cand : candidates) {
            enter_points.push_back(cand.second);
        }
    }

    if (node_level > max_level_) {
        max_level_ = node_level;
        enter_point_ = node_idx;
    }
}

bool HNSWIndex::remove(VectorId id) {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end() || nodes_[it->second].deleted) {
        return false;
    }

    nodes_[it->second].deleted = true;
    --active_count_;
    return true;
}

std::vector<SearchResult> HNSWIndex::search(
    const Vector& query,
    size_t top_k,
    const IdFilterPredicate& filter_fn,
    size_t ef_search
) const {
    if (query.size() != dimension_) {
        throw std::invalid_argument("Query dimension mismatch");
    }
    if (top_k == 0 || active_count_ == 0 || max_level_ < 0) {
        return {};
    }

    size_t ef = (ef_search > 0) ? ef_search : params_.ef_search;
    ef = std::max(ef, top_k);

    size_t curr_obj = enter_point_;
    float curr_dist = distance(query.data(), nodes_[curr_obj].values.data());

    // Greedy descent through upper layers
    for (int l = max_level_; l > 0; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            if (l < static_cast<int>(nodes_[curr_obj].friends.size())) {
                for (size_t neighbor_idx : nodes_[curr_obj].friends[l]) {
                    if (neighbor_idx >= nodes_.size() || nodes_[neighbor_idx].deleted) continue;
                    float d = distance(query.data(), nodes_[neighbor_idx].values.data());
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_obj = neighbor_idx;
                        changed = true;
                    }
                }
            }
        }
    }

    // Search layer 0
    auto candidates = search_layer(query.data(), {curr_obj}, ef, 0);

    std::vector<SearchResult> results;
    results.reserve(top_k);

    for (const auto& cand : candidates) {
        if (results.size() >= top_k) break;
        const auto& node = nodes_[cand.second];
        if (node.deleted) continue;

        if (filter_fn && !filter_fn(node.id)) {
            continue;
        }

        float score = compute_score(query.data(), node.values.data(), dimension_, metric_);
        results.push_back(SearchResult(node.id, score));
    }

    return results;
}

void HNSWIndex::clear() {
    nodes_.clear();
    id_to_index_.clear();
    max_level_ = -1;
    enter_point_ = 0;
    active_count_ = 0;
}

void HNSWIndex::serialize(std::ostream& out) const {
    uint32_t magic = 0x56444248; // "VDBH" (Vector DB HNSW)
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

    uint32_t dim = static_cast<uint32_t>(dimension_);
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

    uint32_t metric_val = static_cast<uint32_t>(metric_);
    out.write(reinterpret_cast<const char*>(&metric_val), sizeof(metric_val));

    out.write(reinterpret_cast<const char*>(&params_.M), sizeof(params_.M));
    out.write(reinterpret_cast<const char*>(&params_.ef_construction), sizeof(params_.ef_construction));
    out.write(reinterpret_cast<const char*>(&params_.ef_search), sizeof(params_.ef_search));

    int32_t max_lvl = max_level_;
    out.write(reinterpret_cast<const char*>(&max_lvl), sizeof(max_lvl));

    uint64_t ep = enter_point_;
    out.write(reinterpret_cast<const char*>(&ep), sizeof(ep));

    uint64_t node_count = nodes_.size();
    out.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));

    for (const auto& node : nodes_) {
        out.write(reinterpret_cast<const char*>(&node.id), sizeof(node.id));
        uint8_t del = node.deleted ? 1 : 0;
        out.write(reinterpret_cast<const char*>(&del), sizeof(del));
        int32_t lvl = node.level;
        out.write(reinterpret_cast<const char*>(&lvl), sizeof(lvl));
        out.write(reinterpret_cast<const char*>(node.values.data()), dimension_ * sizeof(float));

        uint32_t num_layers = static_cast<uint32_t>(node.friends.size());
        out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
        for (uint32_t l = 0; l < num_layers; ++l) {
            uint32_t num_friends = static_cast<uint32_t>(node.friends[l].size());
            out.write(reinterpret_cast<const char*>(&num_friends), sizeof(num_friends));
            if (num_friends > 0) {
                out.write(reinterpret_cast<const char*>(node.friends[l].data()), num_friends * sizeof(size_t));
            }
        }
    }
}

void HNSWIndex::deserialize(std::istream& in) {
    clear();
    uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x56444248) {
        throw std::runtime_error("Invalid HNSWIndex binary header magic");
    }

    uint32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    dimension_ = dim;

    uint32_t metric_val = 0;
    in.read(reinterpret_cast<char*>(&metric_val), sizeof(metric_val));
    metric_ = static_cast<DistanceMetric>(metric_val);

    in.read(reinterpret_cast<char*>(&params_.M), sizeof(params_.M));
    in.read(reinterpret_cast<char*>(&params_.ef_construction), sizeof(params_.ef_construction));
    in.read(reinterpret_cast<char*>(&params_.ef_search), sizeof(params_.ef_search));
    params_.mL = 1.0 / std::log(static_cast<double>(params_.M));

    int32_t max_lvl = 0;
    in.read(reinterpret_cast<char*>(&max_lvl), sizeof(max_lvl));
    max_level_ = max_lvl;

    uint64_t ep = 0;
    in.read(reinterpret_cast<char*>(&ep), sizeof(ep));
    enter_point_ = ep;

    uint64_t node_count = 0;
    in.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));

    nodes_.resize(node_count);
    active_count_ = 0;

    for (size_t i = 0; i < node_count; ++i) {
        auto& node = nodes_[i];
        in.read(reinterpret_cast<char*>(&node.id), sizeof(node.id));
        uint8_t del = 0;
        in.read(reinterpret_cast<char*>(&del), sizeof(del));
        node.deleted = (del != 0);

        int32_t lvl = 0;
        in.read(reinterpret_cast<char*>(&lvl), sizeof(lvl));
        node.level = lvl;

        node.values.resize(dimension_);
        in.read(reinterpret_cast<char*>(node.values.data()), dimension_ * sizeof(float));

        uint32_t num_layers = 0;
        in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
        node.friends.resize(num_layers);
        for (uint32_t l = 0; l < num_layers; ++l) {
            uint32_t num_friends = 0;
            in.read(reinterpret_cast<char*>(&num_friends), sizeof(num_friends));
            node.friends[l].resize(num_friends);
            if (num_friends > 0) {
                in.read(reinterpret_cast<char*>(node.friends[l].data()), num_friends * sizeof(size_t));
            }
        }

        if (!node.deleted) {
            id_to_index_[node.id] = i;
            ++active_count_;
        }
    }
}

} // namespace vectordb
