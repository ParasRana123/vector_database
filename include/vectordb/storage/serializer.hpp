#pragma once

#include "vectordb/core/types.hpp"
#include "vectordb/core/metadata.hpp"
#include <string>
#include <unordered_map>
#include <memory>
#include <iostream>

namespace vectordb {

struct SnapshotData {
    CollectionConfig config;
    std::unordered_map<VectorId, std::string> payloads;
    std::unordered_map<VectorId, Metadata> metadata;
};

class Serializer {
public:
    static constexpr uint32_t SNAPSHOT_MAGIC = 0x56444253; // "VDBS"
    static constexpr uint32_t SNAPSHOT_VERSION = 1;

    static bool save_to_file(
        const std::string& filepath,
        const SnapshotData& snapshot,
        const std::function<void(std::ostream&)>& index_serializer
    );

    static bool load_from_file(
        const std::string& filepath,
        SnapshotData& snapshot,
        const std::function<void(std::istream&)>& index_deserializer
    );

    static void serialize_metadata_map(
        std::ostream& out,
        const std::unordered_map<VectorId, Metadata>& meta_map
    );

    static void deserialize_metadata_map(
        std::istream& in,
        std::unordered_map<VectorId, Metadata>& meta_map
    );
};

} // namespace vectordb
