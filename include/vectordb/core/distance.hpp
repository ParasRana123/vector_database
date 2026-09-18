#pragma once

#include "vectordb/core/types.hpp"
#include <cstddef>
#include <vector>

namespace vectordb {

// Core low-level metric functions operating on raw float pointers
float dot_product(const float* a, const float* b, size_t dim);
float euclidean_distance_sq(const float* a, const float* b, size_t dim);
float euclidean_distance(const float* a, const float* b, size_t dim);
float cosine_similarity(const float* a, const float* b, size_t dim);

// L2 Normalization in-place
void l2_normalize(float* vec, size_t dim);
void l2_normalize(std::vector<float>& vec);

// High-level distance & similarity evaluation
// Distance: lower is closer (e.g. 0.0 means identical)
float compute_distance(const float* a, const float* b, size_t dim, DistanceMetric metric);
float compute_distance(const std::vector<float>& a, const std::vector<float>& b, DistanceMetric metric);

// Similarity score: higher is more similar
float compute_score(const float* a, const float* b, size_t dim, DistanceMetric metric);
float compute_score(const std::vector<float>& a, const std::vector<float>& b, DistanceMetric metric);

} // namespace vectordb
