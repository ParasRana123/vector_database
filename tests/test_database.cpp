#include "vectordb/engine/database.hpp"
#include <iostream>
#include <cassert>
#include <filesystem>

using namespace vectordb;

void test_db_lifecycle() {
    std::string test_dir = "./test_db_data_lifecycle";
    std::filesystem::remove_all(test_dir);

    DatabaseConfig cfg;
    cfg.data_dir = test_dir;
    cfg.enable_wal = true;

    Database db(cfg);

    CollectionConfig c1;
    c1.name = "articles";
    c1.dimension = 4;
    c1.metric = DistanceMetric::COSINE;
    c1.index_type = IndexType::HNSW;

    assert(db.create_collection(c1));
    assert(db.has_collection("articles"));
    assert(!db.has_collection("nonexistent"));

    Metadata m1;
    m1.set_string("topic", "AI");
    m1.set_int("year", 2024);

    db.insert_vector("articles", 1, {1.0f, 0.0f, 0.0f, 0.0f}, "Intro to AI", m1);
    db.insert_vector("articles", 2, {0.0f, 1.0f, 0.0f, 0.0f}, "Quantum Physics", {});

    auto results = db.search_vector("articles", {0.9f, 0.1f, 0.0f, 0.0f}, 1);
    assert(!results.empty());
    assert(results[0].id == 1);
    assert(results[0].payload == "Intro to AI");

    // Search with metadata filter
    auto f = filter::eq("topic", "AI");
    auto filtered_res = db.search_vector("articles", {0.1f, 0.9f, 0.0f, 0.0f}, 5, f);
    assert(!filtered_res.empty());
    assert(filtered_res[0].id == 1);

    std::cout << "[PASS] test_db_lifecycle\n";
    std::filesystem::remove_all(test_dir);
}

void test_db_wal_crash_recovery() {
    std::string test_dir = "./test_db_wal_recovery";
    std::filesystem::remove_all(test_dir);

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

        db.create_collection(c_cfg);
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

        db.create_collection(c_cfg);
        auto coll = db.get_collection("docs");
        assert(coll != nullptr);
        assert(coll->size() == 2);
        assert(coll->contains(101));
        assert(coll->contains(102));

        auto doc = coll->get(101);
        assert(doc.has_value());
        assert(doc->payload == "Document 101");
    }

    std::cout << "[PASS] test_db_wal_crash_recovery\n";
    std::filesystem::remove_all(test_dir);
}

int main() {
    std::cout << "--- Running Database Lifecycle & WAL Tests ---\n";
    test_db_lifecycle();
    test_db_wal_crash_recovery();
    std::cout << "All Database Lifecycle tests passed successfully!\n";
    return 0;
}
