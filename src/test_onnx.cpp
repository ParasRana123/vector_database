#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>

struct Vector {
    uint64_t id;
    std::vector<float> values;
};

struct Document {
    uint64_t id;
    std::string text;
    std::vector<float> embedding;
};

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) {
        throw std::invalid_argument("Cosine similarity requires equally sized, non-empty vectors.");
    }

    float dot = 0.0F;
    float a_squared_norm = 0.0F;
    float b_squared_norm = 0.0F;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        a_squared_norm += a[i] * a[i];
        b_squared_norm += b[i] * b[i];
    }
    const float denominator = std::sqrt(a_squared_norm * b_squared_norm);
    if (denominator == 0.0F) throw std::invalid_argument("Cosine similarity is undefined for a zero vector.");
    return dot / denominator;
}

struct SearchResult {
    uint64_t document_id;
    float score;
};

std::vector<SearchResult> search_documents(const Vector& query, const std::vector<Document>& documents) {
    std::vector<SearchResult> results;
    results.reserve(documents.size());
    for (const auto& document : documents) {
        results.push_back({document.id, cosine_similarity(query.values, document.embedding)});
    }
    std::sort(results.begin(), results.end(), [](const SearchResult& left, const SearchResult& right) {
        return left.score > right.score;
    });
    return results;
}

namespace {
constexpr size_t kMiniLMMaxSequenceLength = 512;

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open tokenizer: " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// all-MiniLM-L6-v2 uses the uncased BERT WordPiece tokenizer.  Loading the
// tokenizer.json keeps its normalization and WordPiece rules in sync with the
// ONNX model rather than trying to reimplement them here.
std::unique_ptr<tokenizers::Tokenizer> load_minilm_tokenizer(const std::filesystem::path& model_dir) {
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(read_file(model_dir / "tokenizer.json"));
    if (tokenizer->TokenToId("[PAD]") != 0 || tokenizer->TokenToId("[UNK]") != 100 ||
        tokenizer->TokenToId("[CLS]") != 101 || tokenizer->TokenToId("[SEP]") != 102) {
        throw std::runtime_error("tokenizer.json is not compatible with all-MiniLM-L6-v2.");
    }
    return tokenizer;
}

std::vector<int64_t> encode_minilm(tokenizers::Tokenizer& tokenizer, const std::string& sentence) {
    // Reserve room for MiniLM's required [CLS] and [SEP] tokens, then truncate
    // to its 512-token BERT context window.
    auto wordpiece_ids = tokenizer.Encode(sentence);
    wordpiece_ids.resize(std::min(wordpiece_ids.size(), kMiniLMMaxSequenceLength - 2));

    std::vector<int64_t> ids;
    ids.reserve(wordpiece_ids.size() + 2);
    ids.push_back(101); // [CLS]
    for (int32_t id : wordpiece_ids) ids.push_back(id);
    ids.push_back(102); // [SEP]
    return ids;
}

std::vector<float> mean_pool_and_normalize(const float* token_embeddings, size_t token_count,
                                            size_t hidden_size, const std::vector<int64_t>& mask) {
    std::vector<float> embedding(hidden_size, 0.0F);
    size_t unmasked_tokens = 0;
    for (size_t token = 0; token < token_count; ++token)
        if (mask[token]) {
            ++unmasked_tokens;
            for (size_t dim = 0; dim < hidden_size; ++dim)
                embedding[dim] += token_embeddings[token * hidden_size + dim];
        }
    float squared_norm = 0.0F;
    for (float& value : embedding) { value /= static_cast<float>(unmasked_tokens); squared_norm += value * value; }
    const float norm = std::sqrt(squared_norm);
    for (float& value : embedding) value /= norm;
    return embedding;
}

std::vector<float> embed_minilm(Ort::Session& session, tokenizers::Tokenizer& tokenizer,
                                 const std::string& sentence, const std::vector<const char*>& input_names,
                                 const char* const* output_names) {
    auto ids = encode_minilm(tokenizer, sentence);
    // Each sentence is run without batch padding, so every token is attended to.
    std::vector<int64_t> mask(ids.size(), 1), type_ids(ids.size(), 0);
    const std::array<int64_t, 2> shape{1, static_cast<int64_t>(ids.size())};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, ids.data(), ids.size(), shape.data(), shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, mask.data(), mask.size(), shape.data(), shape.size()));
    if (input_names.size() == 3) inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, type_ids.data(), type_ids.size(), shape.data(), shape.size()));

    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(), output_names, 1);
    const auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.size() != 3 || output_shape[0] != 1) {
        throw std::runtime_error("Expected output shape [1, tokens, dimensions].");
    }
    return mean_pool_and_normalize(outputs[0].GetTensorData<float>(), static_cast<size_t>(output_shape[1]),
                                   static_cast<size_t>(output_shape[2]), mask);
}
} // namespace

int main() {
    try {
        const auto model_dir = std::filesystem::path(PROJECT_SOURCE_DIR) / "models" / "all-MiniLM-L6-v2";
        auto tokenizer = load_minilm_tokenizer(model_dir);
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MiniLMEmbedding");
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(env, (model_dir / "onnx" / "all-MiniLM-L6-v2.onnx").wstring().c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;

        std::vector<std::string> input_names_owned;
        std::vector<const char*> input_names;
        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            input_names_owned.emplace_back(name.get());
        }
        for (const auto& name : input_names_owned) input_names.push_back(name.c_str());
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        const char* output_names[] = {output_name.get()};

        std::vector<Document> documents = {
            {1, "I study artificial intelligence", {}},
            {2, "I like playing football", {}},
            {3, "Deep learning is a branch of machine learning", {}},
        };
        for (auto& document : documents) {
            document.embedding = embed_minilm(session, *tokenizer, document.text, input_names, output_names);
        }

        const std::string query_text = "machine learning";
        Vector query{0, embed_minilm(session, *tokenizer, query_text, input_names, output_names)};
        const auto results = search_documents(query, documents);
        std::cout << "Query: \"" << query_text << "\"\nRanked documents:\n";
        std::cout << std::fixed << std::setprecision(3);
        for (const auto& result : results) {
            std::cout << result.document_id << " -> " << result.score << '\n';
        }

        std::cout << "MiniLM is ready. Enter a sentence (empty line exits).\n";
        for (std::string sentence; std::cout << "> " && std::getline(std::cin, sentence) && !sentence.empty();) {
            const auto embedding = embed_minilm(session, *tokenizer, sentence, input_names, output_names);
            std::cout << "Embedding (" << embedding.size() << " dimensions): [";
            for (size_t i = 0; i < embedding.size(); ++i) std::cout << (i ? ", " : "") << embedding[i];
            std::cout << "]\n";
        }
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
