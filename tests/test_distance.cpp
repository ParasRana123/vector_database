#include "vectordb/core/distance.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

void test_dot_product() {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {2.0f, 0.0f, 1.0f, -1.0f};
    // 1*2 + 2*0 + 3*1 + 4*(-1) = 2 + 0 + 3 - 4 = 1.0
    float res = vectordb::dot_product(a.data(), b.data(), a.size());
    assert(std::fabs(res - 1.0f) < 1e-5f);
    std::cout << "[PASS] test_dot_product\n";
}

void test_euclidean_distance() {
    std::vector<float> a = {0.0f, 0.0f, 0.0f};
    std::vector<float> b = {3.0f, 4.0f, 0.0f};
    // sqrt(3^2 + 4^2) = 5.0
    float dist = vectordb::euclidean_distance(a.data(), b.data(), a.size());
    assert(std::fabs(dist - 5.0f) < 1e-5f);

    float dist_same = vectordb::euclidean_distance(a.data(), a.data(), a.size());
    assert(std::fabs(dist_same - 0.0f) < 1e-5f);
    std::cout << "[PASS] test_euclidean_distance\n";
}

void test_cosine_similarity() {
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {1.0f, 0.0f, 0.0f};
    std::vector<float> c = {0.0f, 1.0f, 0.0f};
    std::vector<float> d = {-1.0f, 0.0f, 0.0f};

    assert(std::fabs(vectordb::cosine_similarity(a.data(), b.data(), 3) - 1.0f) < 1e-5f);
    assert(std::fabs(vectordb::cosine_similarity(a.data(), c.data(), 3) - 0.0f) < 1e-5f);
    assert(std::fabs(vectordb::cosine_similarity(a.data(), d.data(), 3) - (-1.0f)) < 1e-5f);
    std::cout << "[PASS] test_cosine_similarity\n";
}

void test_normalization() {
    std::vector<float> v = {3.0f, 4.0f, 0.0f};
    vectordb::l2_normalize(v);
    assert(std::fabs(v[0] - 0.6f) < 1e-5f);
    assert(std::fabs(v[1] - 0.8f) < 1e-5f);
    assert(std::fabs(v[2] - 0.0f) < 1e-5f);

    float norm = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    assert(std::fabs(norm - 1.0f) < 1e-5f);
    std::cout << "[PASS] test_normalization\n";
}

void test_simd_long_vectors() {
    const size_t dim = 384;
    std::vector<float> a(dim, 1.0f);
    std::vector<float> b(dim, 2.0f);

    float dot = vectordb::dot_product(a.data(), b.data(), dim);
    assert(std::fabs(dot - static_cast<float>(dim * 2)) < 1e-3f);

    float l2 = vectordb::euclidean_distance(a.data(), b.data(), dim);
    float expected_l2 = std::sqrt(static_cast<float>(dim));
    assert(std::fabs(l2 - expected_l2) < 1e-3f);
    std::cout << "[PASS] test_simd_long_vectors\n";
}

int main() {
    std::cout << "--- Running Distance Metric Tests ---\n";
    test_dot_product();
    test_euclidean_distance();
    test_cosine_similarity();
    test_normalization();
    test_simd_long_vectors();
    std::cout << "All distance metric tests passed successfully!\n";
    return 0;
}
