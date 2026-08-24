#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace {
constexpr size_t kMaxSequenceLength = 256;

class BertTokenizer {
public:
    explicit BertTokenizer(const std::filesystem::path& vocab_path) {
        std::ifstream file(vocab_path);
        if (!file) throw std::runtime_error("Cannot open vocabulary: " + vocab_path.string());
        std::string token;
        for (int64_t id = 0; std::getline(file, token); ++id) vocab_[token] = id;
        for (const char* required : {"[PAD]", "[UNK]", "[CLS]", "[SEP]"})
            if (!vocab_.count(required)) throw std::runtime_error("Vocabulary is missing " + std::string(required));
    }

    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<int64_t> ids{vocab_.at("[CLS]")};
        for (const auto& word : basic_tokenize(text)) {
            for (const auto& piece : wordpiece_tokenize(word)) {
                if (ids.size() + 1 >= kMaxSequenceLength) break;
                ids.push_back(vocab_.at(piece));
            }
            if (ids.size() + 1 >= kMaxSequenceLength) break;
        }
        ids.push_back(vocab_.at("[SEP]"));
        return ids;
    }

private:
    std::unordered_map<std::string, int64_t> vocab_;

    static std::vector<std::string> basic_tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::string current;
        for (unsigned char c : text) {
            if (std::isalnum(c)) current.push_back(static_cast<char>(std::tolower(c)));
            else {
                if (!current.empty()) { tokens.push_back(current); current.clear(); }
                if (std::ispunct(c)) tokens.emplace_back(1, static_cast<char>(c));
            }
        }
        if (!current.empty()) tokens.push_back(current);
        return tokens;
    }

    std::vector<std::string> wordpiece_tokenize(const std::string& word) const {
        std::vector<std::string> pieces;
        for (size_t start = 0; start < word.size();) {
            size_t end = word.size();
            std::string found;
            while (end > start) {
                std::string candidate = (start ? "##" : "") + word.substr(start, end - start);
                if (vocab_.count(candidate)) { found = std::move(candidate); break; }
                --end;
            }
            if (found.empty()) return {"[UNK]"};
            pieces.push_back(std::move(found));
            start = end;
        }
        return pieces;
    }
};

std::vector<float> mean_pool_and_normalize(const float* token_embeddings, size_t token_count,
                                            size_t hidden_size, const std::vector<int64_t>& mask) {
    std::vector<float> embedding(hidden_size, 0.0F);
    for (size_t token = 0; token < token_count; ++token)
        if (mask[token]) for (size_t dim = 0; dim < hidden_size; ++dim)
            embedding[dim] += token_embeddings[token * hidden_size + dim];
    float squared_norm = 0.0F;
    for (float& value : embedding) { value /= static_cast<float>(token_count); squared_norm += value * value; }
    const float norm = std::sqrt(squared_norm);
    for (float& value : embedding) value /= norm;
    return embedding;
}
} // namespace

int main() {
    try {
        const auto model_dir = std::filesystem::path(PROJECT_SOURCE_DIR) / "models" / "all-MiniLM-L6-v2";
        BertTokenizer tokenizer(model_dir / "vocab.txt");
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

        std::cout << "MiniLM is ready. Enter a sentence (empty line exits).\n";
        for (std::string sentence; std::cout << "> " && std::getline(std::cin, sentence) && !sentence.empty();) {
            std::vector<int64_t> ids = tokenizer.encode(sentence);
            std::vector<int64_t> mask(ids.size(), 1), type_ids(ids.size(), 0);
            const std::array<int64_t, 2> shape{1, static_cast<int64_t>(ids.size())};
            auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<Ort::Value> inputs;
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, ids.data(), ids.size(), shape.data(), shape.size()));
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, mask.data(), mask.size(), shape.data(), shape.size()));
            if (input_names.size() == 3) inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, type_ids.data(), type_ids.size(), shape.data(), shape.size()));

            auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(), output_names, 1);
            const auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            if (output_shape.size() != 3 || output_shape[0] != 1) throw std::runtime_error("Expected output shape [1, tokens, dimensions].");
            const size_t tokens = static_cast<size_t>(output_shape[1]);
            const size_t dimensions = static_cast<size_t>(output_shape[2]);
            const auto embedding = mean_pool_and_normalize(outputs[0].GetTensorData<float>(), tokens, dimensions, mask);
            std::cout << "Embedding (" << dimensions << " dimensions): [";
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
