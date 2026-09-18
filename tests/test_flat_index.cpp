#include "vectordb/index/flat_index.hpp"
#include <iostream>
#include <cassert>
#include <sstream>

using namespace vectordb;

void test_flat_index_crud() {
    FlatIndex index(3, DistanceMetric::EUCLIDEAN);
    assert(index.size() == 0);

    index.add(1, {1.0f, 0.0f, 0.0f});
    index.add(2, {0.0f, 2.0f, 0.0f});
    index.add(3, {0.0f, 0.0f, 3.0f});

    assert(index.size() == 3);
    assert(index.contains(2));
    assert(!index.contains(4));

    auto v = index.get_vector(1);
    assert(v.has_value());
    assert((*v)[0] == 1.0f && (*v)[1] == 0.0f && (*v)[2] == 0.0f);

    bool removed = index.remove(2);
    assert(removed);
    assert(index.size() == 2);
    assert(!index.contains(2));
    assert(index.contains(3));

    std::cout << "[PASS] test_flat_index_crud\n";
}

void test_flat_index_search() {
    FlatIndex index(2, DistanceMetric::COSINE);
    index.add(10, {1.0f, 0.0f});
    index.add(20, {0.7071f, 0.7071f});
    index.add(30, {0.0f, 1.0f});
    index.add(40, {-1.0f, 0.0f});

    Vector query = {1.0f, 0.1f};
    auto results = index.search(query, 2);

    assert(results.size() == 2);
    assert(results[0].id == 10);
    assert(results[1].id == 20);

    // Search with filter
    auto filtered = index.search(query, 5, [](VectorId id) {
        return id != 10;
    });
    assert(!filtered.empty());
    assert(filtered[0].id == 20);

    std::cout << "[PASS] test_flat_index_search\n";
}

void test_flat_index_serialization() {
    FlatIndex original(3, DistanceMetric::DOT_PRODUCT);
    original.add(100, {0.5f, 1.5f, 2.5f});
    original.add(200, {1.0f, 2.0f, 3.0f});

    std::stringstream ss;
    original.serialize(ss);

    FlatIndex loaded(1, DistanceMetric::COSINE); // dummy init
    loaded.deserialize(ss);

    assert(loaded.size() == 2);
    assert(loaded.dimension() == 3);
    assert(loaded.metric() == DistanceMetric::DOT_PRODUCT);
    assert(loaded.contains(100));
    assert(loaded.contains(200));

    auto res = loaded.search({1.0f, 2.0f, 3.0f}, 1);
    assert(!res.empty());
    assert(res[0].id == 200);

    std::cout << "[PASS] test_flat_index_serialization\n";
}

int main() {
    std::cout << "--- Running Flat Index Tests ---\n";
    test_flat_index_crud();
    test_flat_index_search();
    test_flat_index_serialization();
    std::cout << "All Flat Index tests passed successfully!\n";
    return 0;
}
