#include "vectordb/embedding/embedder.hpp"
#include "vectordb/embedding/tokenizer.hpp"
#include "vectordb/core/distance.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <array>
#include <random>
#include <mutex>

#include <onnxruntime_cxx_api.h>

namespace vectordb {

struct TextEmbedder::Impl {
    BertTokenizer tokenizer;
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names_owned;
    std::vector<const char*> input_names;
    std::string output_name_owned;
    std::vector<const char*> output_names;
};

TextEmbedder::TextEmbedder() : impl_(std::make_unique<Impl>()) {}
TextEmbedder::~TextEmbedder() = default;

bool TextEmbedder::load(const std::string& model_dir_str) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        std::filesystem::path model_dir(model_dir_str);
        auto vocab_path = model_dir / "vocab.txt";
        auto tokenizer_path = model_dir / "tokenizer.json";
        auto onnx_path = model_dir / "onnx" / "all-MiniLM-L6-v2.onnx";

        if (!std::filesystem::exists(onnx_path)) {
            onnx_path = model_dir / "all-MiniLM-L6-v2.onnx";
            if (!std::filesystem::exists(onnx_path)) {
                std::cerr << "[TextEmbedder] Warning: ONNX model not found in " << model_dir_str << "\n";
                is_loaded_ = false;
                return false;
            }
        }

        bool tok_loaded = false;
        if (std::filesystem::exists(vocab_path)) {
            tok_loaded = impl_->tokenizer.load_vocab_file(vocab_path.string());
        } else if (std::filesystem::exists(tokenizer_path)) {
            tok_loaded = impl_->tokenizer.load_json_file(tokenizer_path.string());
        }

        if (!tok_loaded) {
            std::cerr << "[TextEmbedder] Warning: Could not load vocab or tokenizer.json in " << model_dir_str << "\n";
            is_loaded_ = false;
            return false;
        }

        impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "VectorDBEmbedder");

        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#if defined(_WIN32)
        impl_->session = std::make_unique<Ort::Session>(*impl_->env, onnx_path.wstring().c_str(), options);
#else
        impl_->session = std::make_unique<Ort::Session>(*impl_->env, onnx_path.string().c_str(), options);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        impl_->input_names_owned.clear();
        impl_->input_names.clear();
        for (size_t i = 0; i < impl_->session->GetInputCount(); ++i) {
            auto name = impl_->session->GetInputNameAllocated(i, allocator);
            impl_->input_names_owned.emplace_back(name.get());
        }
        for (const auto& name : impl_->input_names_owned) {
            impl_->input_names.push_back(name.c_str());
        }

        auto out_name = impl_->session->GetOutputNameAllocated(0, allocator);
        impl_->output_name_owned = out_name.get();
        impl_->output_names = {impl_->output_name_owned.c_str()};

        dimension_ = 384;
        is_loaded_ = true;
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[TextEmbedder] Load error: " << ex.what() << "\n";
        is_loaded_ = false;
        return false;
    }
}

static Vector generate_fallback_embedding(const std::string& text, size_t dim) {
    Vector vec(dim, 0.0f);
    std::hash<std::string> hasher;
    size_t seed = hasher(text);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
    l2_normalize(vec);
    return vec;
}

Vector TextEmbedder::embed(const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_loaded_ || !impl_->session || !impl_->tokenizer.is_loaded()) {
        return generate_fallback_embedding(text, dimension_);
    }

    try {
        auto ids = impl_->tokenizer.encode(text, 512);

        std::vector<int64_t> mask(ids.size(), 1);
        std::vector<int64_t> type_ids(ids.size(), 0);
        const std::array<int64_t, 2> shape{1, static_cast<int64_t>(ids.size())};

        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, ids.data(), ids.size(), shape.data(), shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, mask.data(), mask.size(), shape.data(), shape.size()));
        if (impl_->input_names.size() == 3) {
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, type_ids.data(), type_ids.size(), shape.data(), shape.size()));
        }

        auto outputs = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_names.data(),
            inputs.data(),
            inputs.size(),
            impl_->output_names.data(),
            1
        );

        const auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (output_shape.size() != 3 || output_shape[0] != 1) {
            return generate_fallback_embedding(text, dimension_);
        }

        size_t token_count = static_cast<size_t>(output_shape[1]);
        size_t hidden_size = static_cast<size_t>(output_shape[2]);
        const float* token_embeddings = outputs[0].GetTensorData<float>();

        // Mean pooling & normalization
        Vector embedding(hidden_size, 0.0f);
        size_t unmasked_tokens = 0;
        for (size_t token = 0; token < token_count; ++token) {
            if (mask[token]) {
                ++unmasked_tokens;
                for (size_t d = 0; d < hidden_size; ++d) {
                    embedding[d] += token_embeddings[token * hidden_size + d];
                }
            }
        }
        if (unmasked_tokens > 0) {
            for (float& val : embedding) {
                val /= static_cast<float>(unmasked_tokens);
            }
        }
        l2_normalize(embedding);
        return embedding;
    } catch (const std::exception& ex) {
        std::cerr << "[TextEmbedder] Inference error: " << ex.what() << "\n";
        return generate_fallback_embedding(text, dimension_);
    }
}

std::vector<Vector> TextEmbedder::embed_batch(const std::vector<std::string>& texts) {
    std::vector<Vector> results;
    results.reserve(texts.size());
    for (const auto& text : texts) {
        results.push_back(embed(text));
    }
    return results;
}

} // namespace vectordb
