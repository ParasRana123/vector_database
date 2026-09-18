#include "vectordb/core/metadata.hpp"
#include <iostream>
#include <cassert>

using namespace vectordb;

void test_metadata_basic() {
    Metadata meta;
    meta.set_string("title", "Quantum Computing");
    meta.set_int("year", 2024);
    meta.set_double("score", 98.5);
    meta.set_bool("is_published", true);

    assert(meta.has("title"));
    assert(meta.has("year"));
    assert(!meta.has("nonexistent"));

    assert(meta.get_string("title").value() == "Quantum Computing");
    assert(meta.get_int("year").value() == 2024);
    assert(meta.get_double("score").value() == 98.5);
    assert(meta.get_bool("is_published").value() == true);

    std::cout << "[PASS] test_metadata_basic\n";
}

void test_metadata_json_roundtrip() {
    Metadata meta;
    meta.set_string("author", "Alice");
    meta.set_int("views", 1500);
    meta.set_bool("featured", false);

    std::string json = meta.to_json();
    Metadata restored = Metadata::from_json(json);

    assert(restored.get_string("author").value() == "Alice");
    assert(restored.get_int("views").value() == 1500);
    assert(restored.get_bool("featured").value() == false);

    std::cout << "[PASS] test_metadata_json_roundtrip\n";
}

void test_metadata_filters() {
    Metadata doc1;
    doc1.set_string("category", "physics");
    doc1.set_int("year", 2021);
    doc1.set_double("price", 29.99);

    Metadata doc2;
    doc2.set_string("category", "biology");
    doc2.set_int("year", 2023);
    doc2.set_double("price", 49.99);

    auto f_cat = filter::eq("category", std::string("physics"));
    assert(f_cat->matches(doc1) == true);
    assert(f_cat->matches(doc2) == false);

    auto f_year = filter::gte("year", int64_t(2022));
    assert(f_year->matches(doc1) == false);
    assert(f_year->matches(doc2) == true);

    auto f_in = filter::in("category", {std::string("physics"), std::string("chemistry"), std::string("math")});
    assert(f_in->matches(doc1) == true);
    assert(f_in->matches(doc2) == false);

    // AND composite filter
    auto f_and = filter::all_of({
        filter::eq("category", std::string("biology")),
        filter::gt("price", 30.0)
    });
    assert(f_and->matches(doc1) == false);
    assert(f_and->matches(doc2) == true);

    // OR composite filter
    auto f_or = filter::any_of({
        filter::eq("category", std::string("physics")),
        filter::eq("category", std::string("biology"))
    });
    assert(f_or->matches(doc1) == true);
    assert(f_or->matches(doc2) == true);

    // NOT filter
    auto f_not = filter::not_filter(filter::eq("category", std::string("physics")));
    assert(f_not->matches(doc1) == false);
    assert(f_not->matches(doc2) == true);

    std::cout << "[PASS] test_metadata_filters\n";
}

int main() {
    std::cout << "--- Running Metadata & Filter Tests ---\n";
    test_metadata_basic();
    test_metadata_json_roundtrip();
    test_metadata_filters();
    std::cout << "All metadata and filter tests passed successfully!\n";
    return 0;
}
