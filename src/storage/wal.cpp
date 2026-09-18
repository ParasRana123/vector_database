#include "vectordb/storage/wal.hpp"
#include <filesystem>
#include <iostream>
#include <mutex>

namespace vectordb {

static void write_wal_str(std::ofstream& out, const std::string& str) {
    uint32_t len = static_cast<uint32_t>(str.size());
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    if (len > 0) {
        out.write(str.data(), len);
    }
}

static std::string read_wal_str(std::ifstream& in) {
    uint32_t len = 0;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (len == 0 || in.fail()) return "";
    std::string s(len, '\0');
    in.read(&s[0], len);
    return s;
}

WAL::WAL(std::string wal_path) : path_(std::move(wal_path)) {}

WAL::~WAL() {
    close();
}

void WAL::write_header() {
    uint32_t magic = WAL_MAGIC;
    out_.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    uint32_t ver = WAL_VERSION;
    out_.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
}

bool WAL::open() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_open_) return true;

    bool exists = std::filesystem::exists(path_) && (std::filesystem::file_size(path_) > 0);
    out_.open(path_, std::ios::binary | std::ios::app);
    if (!out_) return false;

    is_open_ = true;
    if (!exists) {
        write_header();
        out_.flush();
    }
    return true;
}

void WAL::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_open_) {
        if (out_.is_open()) {
            out_.flush();
            out_.close();
        }
        is_open_ = false;
    }
}

void WAL::append_insert(VectorId id, const Vector& vector, const std::string& payload, const Metadata& metadata) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_open_) open();

    uint8_t op = static_cast<uint8_t>(WALOpType::INSERT);
    out_.write(reinterpret_cast<const char*>(&op), sizeof(op));
    out_.write(reinterpret_cast<const char*>(&id), sizeof(id));

    uint32_t dim = static_cast<uint32_t>(vector.size());
    out_.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    if (dim > 0) {
        out_.write(reinterpret_cast<const char*>(vector.data()), dim * sizeof(float));
    }

    write_wal_str(out_, payload);
    write_wal_str(out_, metadata.to_json());
    out_.flush();
}

void WAL::append_delete(VectorId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_open_) open();

    uint8_t op = static_cast<uint8_t>(WALOpType::DELETE);
    out_.write(reinterpret_cast<const char*>(&op), sizeof(op));
    out_.write(reinterpret_cast<const char*>(&id), sizeof(id));
    out_.flush();
}

void WAL::append_clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_open_) open();

    uint8_t op = static_cast<uint8_t>(WALOpType::CLEAR);
    out_.write(reinterpret_cast<const char*>(&op), sizeof(op));
    out_.flush();
}

void WAL::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_open_ && out_.is_open()) {
        out_.flush();
    }
}

void WAL::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (out_.is_open()) {
        out_.close();
    }
    out_.open(path_, std::ios::binary | std::ios::trunc);
    if (out_.is_open()) {
        write_header();
        out_.flush();
        is_open_ = true;
    }
}

size_t WAL::replay(const std::string& wal_path, const ReplayCallback& callback) {
    if (!std::filesystem::exists(wal_path)) return 0;

    std::ifstream in(wal_path, std::ios::binary);
    if (!in) return 0;

    uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != WAL_MAGIC) return 0;

    uint32_t ver = 0;
    in.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    if (ver > WAL_VERSION) return 0;

    size_t count = 0;
    uint8_t op_byte = 0;
    while (in.read(reinterpret_cast<char*>(&op_byte), sizeof(op_byte))) {
        WALRecord rec;
        rec.type = static_cast<WALOpType>(op_byte);

        if (rec.type == WALOpType::INSERT) {
            in.read(reinterpret_cast<char*>(&rec.id), sizeof(rec.id));
            uint32_t dim = 0;
            in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            rec.vector.resize(dim);
            if (dim > 0) {
                in.read(reinterpret_cast<char*>(rec.vector.data()), dim * sizeof(float));
            }
            rec.payload = read_wal_str(in);
            std::string meta_json = read_wal_str(in);
            rec.metadata = Metadata::from_json(meta_json);
        } else if (rec.type == WALOpType::DELETE) {
            in.read(reinterpret_cast<char*>(&rec.id), sizeof(rec.id));
        } else if (rec.type == WALOpType::CLEAR) {
            // No extra payload
        }

        if (!in.fail() && callback) {
            callback(rec);
            ++count;
        }
    }
    return count;
}

} // namespace vectordb
