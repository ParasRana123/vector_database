#include "vectordb/storage/serializer.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <functional>
namespace vectordb {

static void write_string(std::ostream& out, const std::string& str) {
    uint32_t len = static_cast<uint32_t>(str.size());
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    if (len > 0) {
        out.write(str.data(), len);
    }
}

static std::string read_string(std::istream& in) {
    uint32_t len = 0;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (len == 0) return "";
    std::string str(len, '\0');
    in.read(&str[0], len);
    return str;
}

void Serializer::serialize_metadata_map(
    std::ostream& out,
    const std::unordered_map<VectorId, Metadata>& meta_map
) {
    uint64_t count = meta_map.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& [id, meta] : meta_map) {
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        std::string json_str = meta.to_json();
        write_string(out, json_str);
    }
}

void Serializer::deserialize_metadata_map(
    std::istream& in,
    std::unordered_map<VectorId, Metadata>& meta_map
) {
    meta_map.clear();
    uint64_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    for (size_t i = 0; i < count; ++i) {
        VectorId id = 0;
        in.read(reinterpret_cast<char*>(&id), sizeof(id));
        std::string json_str = read_string(in);
        meta_map[id] = Metadata::from_json(json_str);
    }
}

bool Serializer::save_to_file(
    const std::string& filepath,
    const SnapshotData& snapshot,
    const std::function<void(std::ostream&)>& index_serializer
) {
    std::ofstream out(filepath, std::ios::binary | std::ios::trunc);
    if (!out) return false;

    // Header
    uint32_t magic = SNAPSHOT_MAGIC;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

    uint32_t version = SNAPSHOT_VERSION;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Config
    write_string(out, snapshot.config.name);
    uint32_t dim = static_cast<uint32_t>(snapshot.config.dimension);
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

    uint32_t metric_val = static_cast<uint32_t>(snapshot.config.metric);
    out.write(reinterpret_cast<const char*>(&metric_val), sizeof(metric_val));

    uint32_t index_type_val = static_cast<uint32_t>(snapshot.config.index_type);
    out.write(reinterpret_cast<const char*>(&index_type_val), sizeof(index_type_val));

    // Payloads
    uint64_t payload_count = snapshot.payloads.size();
    out.write(reinterpret_cast<const char*>(&payload_count), sizeof(payload_count));
    for (const auto& [id, payload] : snapshot.payloads) {
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        write_string(out, payload);
    }

    // Metadata
    serialize_metadata_map(out, snapshot.metadata);

    // Index Payload
    if (index_serializer) {
        index_serializer(out);
    }

    return out.good();
}

bool Serializer::load_from_file(
    const std::string& filepath,
    SnapshotData& snapshot,
    const std::function<void(std::istream&)>& index_deserializer
) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) return false;

    uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != SNAPSHOT_MAGIC) return false;

    uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version > SNAPSHOT_VERSION) return false;

    snapshot.config.name = read_string(in);
    uint32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    snapshot.config.dimension = dim;

    uint32_t metric_val = 0;
    in.read(reinterpret_cast<char*>(&metric_val), sizeof(metric_val));
    snapshot.config.metric = static_cast<DistanceMetric>(metric_val);

    uint32_t index_type_val = 0;
    in.read(reinterpret_cast<char*>(&index_type_val), sizeof(index_type_val));
    snapshot.config.index_type = static_cast<IndexType>(index_type_val);

    // Payloads
    snapshot.payloads.clear();
    uint64_t payload_count = 0;
    in.read(reinterpret_cast<char*>(&payload_count), sizeof(payload_count));
    for (size_t i = 0; i < payload_count; ++i) {
        VectorId id = 0;
        in.read(reinterpret_cast<char*>(&id), sizeof(id));
        snapshot.payloads[id] = read_string(in);
    }

    // Metadata
    deserialize_metadata_map(in, snapshot.metadata);

    // Index Payload
    if (index_deserializer) {
        index_deserializer(in);
    }

    return in.good();
}

} // namespace vectordb
