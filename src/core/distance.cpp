#include "vectordb/core/distance.hpp"
#include <cmath>
#include <stdexcept>

#if defined(_MSC_VER)
#include <immintrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace vectordb {

float dot_product(const float* a, const float* b, size_t dim) {
    if (!a || !b) return 0.0f;

    size_t i = 0;
    float sum = 0.0f;

#if defined(__AVX2__)
    __m256 sum256 = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum256 = _mm256_fmadd_ps(va, vb, sum256);
    }
    // Horizontal add of 8 floats
    alignas(32) float buffer[8];
    _mm256_storeu_ps(buffer, sum256);
    sum = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
#endif

    // Scalar fallback or remainder
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float euclidean_distance_sq(const float* a, const float* b, size_t dim) {
    if (!a || !b) return 0.0f;

    size_t i = 0;
    float sum = 0.0f;

#if defined(__AVX2__)
    __m256 sum256 = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum256 = _mm256_fmadd_ps(diff, diff, sum256);
    }
    alignas(32) float buffer[8];
    _mm256_storeu_ps(buffer, sum256);
    sum = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
#endif

    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float euclidean_distance(const float* a, const float* b, size_t dim) {
    return std::sqrt(std::max(0.0f, euclidean_distance_sq(a, b, dim)));
}

float cosine_similarity(const float* a, const float* b, size_t dim) {
    if (!a || !b || dim == 0) return 0.0f;

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom <= 1e-12f) return 0.0f;
    
    float sim = dot / denom;
    // Clamp to [-1.0, 1.0] to handle floating point precision edge cases
    if (sim > 1.0f) sim = 1.0f;
    if (sim < -1.0f) sim = -1.0f;
    return sim;
}

void l2_normalize(float* vec, size_t dim) {
    if (!vec || dim == 0) return;
    float norm_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        norm_sq += vec[i] * vec[i];
    }
    float norm = std::sqrt(norm_sq);
    if (norm > 1e-12f) {
        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < dim; ++i) {
            vec[i] *= inv_norm;
        }
    }
}

void l2_normalize(std::vector<float>& vec) {
    l2_normalize(vec.data(), vec.size());
}

float compute_distance(const float* a, const float* b, size_t dim, DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::EUCLIDEAN:
            return euclidean_distance(a, b, dim);
        case DistanceMetric::COSINE:
            // Cosine distance = 1.0 - cosine_similarity (range [0, 2])
            return 1.0f - cosine_similarity(a, b, dim);
        case DistanceMetric::DOT_PRODUCT:
            // Negative dot product so lower distance = higher dot product
            return -dot_product(a, b, dim);
        default:
            return euclidean_distance(a, b, dim);
    }
}

float compute_distance(const std::vector<float>& a, const std::vector<float>& b, DistanceMetric metric) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions mismatch: " + std::to_string(a.size()) + " vs " + std::to_string(b.size()));
    }
    return compute_distance(a.data(), b.data(), a.size(), metric);
}

float compute_score(const float* a, const float* b, size_t dim, DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::COSINE:
            return cosine_similarity(a, b, dim);
        case DistanceMetric::DOT_PRODUCT:
            return dot_product(a, b, dim);
        case DistanceMetric::EUCLIDEAN: {
            float dist = euclidean_distance(a, b, dim);
            return 1.0f / (1.0f + dist); // Convert L2 distance into [0, 1] similarity
        }
        default:
            return cosine_similarity(a, b, dim);
    }
}

float compute_score(const std::vector<float>& a, const std::vector<float>& b, DistanceMetric metric) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions mismatch: " + std::to_string(a.size()) + " vs " + std::to_string(b.size()));
    }
    return compute_score(a.data(), b.data(), a.size(), metric);
}

} // namespace vectordb
