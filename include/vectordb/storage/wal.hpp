#pragma once

#include "vectordb/core/types.hpp"
#include "vectordb/core/metadata.hpp"
#include <string>
#include <fstream>
#include <mutex>
#include <functional>

namespace vectordb {

enum class WALOpType : uint8_t {
    INSERT = 1,
    DELETE = 2,
    CLEAR = 3
};

struct WALRecord {
    WALOpType type;
    VectorId id{0};
    Vector vector;
    std::string payload;
    Metadata metadata;
};

class WAL {
public:
    static constexpr uint32_t WAL_MAGIC = 0x56444257; // "VDBW"
    static constexpr uint32_t WAL_VERSION = 1;

    explicit WAL(std::string wal_path);
    ~WAL();

    bool open();
    void close();

    void append_insert(VectorId id, const Vector& vector, const std::string& payload, const Metadata& metadata);
    void append_delete(VectorId id);
    void append_clear();

    void flush();
    void clear();

    using ReplayCallback = std::function<void(const WALRecord&)>;
    static size_t replay(const std::string& wal_path, const ReplayCallback& callback);

private:
    std::string path_;
    std::ofstream out_;
    std::mutex mutex_;
    bool is_open_{false};

    void write_header();
};

} // namespace vectordb
