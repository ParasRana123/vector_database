#pragma once

#include "vectordb/core/types.hpp"
#include <string>
#include <vector>
#include <memory>
#include <mutex>

namespace vectordb {

class TextEmbedder {
public:
    TextEmbedder();
    ~TextEmbedder();

    // Load MiniLM model and tokenizer from model directory
    bool load(const std::string& model_dir);

    // Vectorize single sentence
    Vector embed(const std::string& text);

    // Batch vectorization
    std::vector<Vector> embed_batch(const std::vector<std::string>& texts);

    size_t dimension() const { return dimension_; }
    bool is_loaded() const { return is_loaded_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    size_t dimension_{384};
    bool is_loaded_{false};
    std::mutex mutex_;
};

} // namespace vectordb
