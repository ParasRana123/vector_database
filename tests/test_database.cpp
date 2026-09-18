#include "vectordb/engine/database.hpp"
#include <iostream>
#include <filesystem>
#include <cstdlib>

#define DB_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "[ASSERTION FAILED] " << #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

using namespace vectordb;

void test_db_lifecycle() {
    std::string test_dir = "./test_db_data_lifecycle";
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);

    {
        DatabaseConfig cfg;
        cfg.data_dir = test_dir;
        cfg.enable_wal = true;

        Database db(cfg);

        CollectionConfig c1;
        c1.name = "articles";
        c1.dimension = 4;
        c1.metric = DistanceMetric::COSINE;
        c1.index_type = IndexType::HNSW;

        bool created = db.create_collection(c1);
        DB_ASSERT(created);
        DB_ASSERT(db.has_collection("articles"));
        DB_ASSERT(!db.has_collection("nonexistent"));

        Metadata m1;
        m1.set_string("topic", "AI");
        m1.set_int("year", 2024);

        db.insert_vector("articles", 1, {1.0f, 0.0f, 0.0f, 0.0f}, "Intro to AI", m1);
        db.insert_vector("articles", 2, {0.0f, 1.0f, 0.0f, 0.0f}, "Quantum Physics", {});

        auto results = db.search_vector("articles", {0.9f, 0.1f, 0.0f, 0.0f}, 1);
        DB_ASSERT(!results.empty());
        DB_ASSERT(results[0].id == 1);
        DB_ASSERT(results[0].payload == "Intro to AI");

        // Search with metadata filter
        auto f = filter::eq("topic", std::string("AI"));
        auto filtered_res = db.search_vector("articles", {0.1f, 0.9f, 0.0f, 0.0f}, 5, f);
        DB_ASSERT(!filtered_res.empty());
        DB_ASSERT(filtered_res[0].id == 1);
    } // db destroyed, files closed

    std::filesystem::remove_all(test_dir, ec);
    std::cout << "[PASS] test_db_lifecycle\n";
}

void test_db_wal_crash_recovery() {
    std::string test_dir = "./test_db_wal_recovery";
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);

    {
        // 1. First DB instance: write data and let it flush to WAL
        DatabaseConfig cfg;
        cfg.data_dir = test_dir;
        cfg.enable_wal = true;

        Database db(cfg);
        CollectionConfig c_cfg;
        c_cfg.name = "docs";
        c_cfg.dimension = 3;
        c_cfg.metric = DistanceMetric::EUCLIDEAN;
        c_cfg.index_type = IndexType::FLAT;

        bool created = db.create_collection(c_cfg);
        DB_ASSERT(created);
        db.insert_vector("docs", 101, {1.0f, 2.0f, 3.0f}, "Document 101");
        db.insert_vector("docs", 102, {4.0f, 5.0f, 6.0f}, "Document 102");
    } // db destroyed, simulating shutdown/restart

    {
        // 2. New DB instance on same folder: should replay WAL
        DatabaseConfig cfg;
        cfg.data_dir = test_dir;
        cfg.enable_wal = true;

        Database db(cfg);
        CollectionConfig c_cfg;
        c_cfg.name = "docs";
        c_cfg.dimension = 3;
        c_cfg.metric = DistanceMetric::EUCLIDEAN;
        c_cfg.index_type = IndexType::FLAT;

        bool created = db.create_collection(c_cfg);
        DB_ASSERT(created);
        auto coll = db.get_collection("docs");
        DB_ASSERT(coll != nullptr);
        DB_ASSERT(coll->size() == 2);
        DB_ASSERT(coll->contains(101));
        DB_ASSERT(coll->contains(102));

        auto doc = coll->get(101);
        DB_ASSERT(doc.has_value());
        DB_ASSERT(doc->payload == "Document 101");
    }

    std::filesystem::remove_all(test_dir, ec);
    std::cout << "[PASS] test_db_wal_crash_recovery\n";
}

int main() {
    try {
        std::cout << "--- Running Database Lifecycle & WAL Tests ---\n";
        test_db_lifecycle();
        test_db_wal_crash_recovery();
        std::cout << "All Database Lifecycle tests passed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
