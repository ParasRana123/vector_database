#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>

namespace vectordb {

class BertTokenizer {
public:
    BertTokenizer() = default;

    bool load_vocab_file(const std::string& vocab_path) {
        std::ifstream in(vocab_path);
        if (!in) return false;

        vocab_.clear();
        std::string line;
        int64_t id = 0;
        while (std::getline(in, line)) {
            // Trim carriage returns if on Windows CRLF
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) {
                vocab_[line] = id;
            }
            ++id;
        }

        init_special_tokens();
        return !vocab_.empty();
    }

    bool load_json_file(const std::string& json_path) {
        std::ifstream in(json_path);
        if (!in) return false;

        std::string content((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());

        vocab_.clear();
        // Extract "vocab": { ... } mapping from tokenizer.json
        size_t vocab_pos = content.find("\"vocab\"");
        if (vocab_pos == std::string::npos) return false;

        size_t start_brace = content.find('{', vocab_pos);
        if (start_brace == std::string::npos) return false;

        size_t pos = start_brace + 1;
        while (pos < content.size()) {
            // Skip whitespace
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' || content[pos] == '\n' || content[pos] == '\r' || content[pos] == ',')) {
                ++pos;
            }
            if (pos >= content.size() || content[pos] == '}') break;

            if (content[pos] != '"') {
                ++pos;
                continue;
            }

            // Read token key
            ++pos;
            std::string key;
            while (pos < content.size() && content[pos] != '"') {
                if (content[pos] == '\\' && pos + 1 < content.size()) {
                    ++pos;
                    key += content[pos];
                } else {
                    key += content[pos];
                }
                ++pos;
            }
            if (pos < content.size() && content[pos] == '"') ++pos;

            // Skip to colon
            while (pos < content.size() && content[pos] != ':') ++pos;
            if (pos < content.size()) ++pos;

            // Skip whitespace
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) ++pos;

            // Read integer id
            size_t num_start = pos;
            while (pos < content.size() && (std::isdigit(static_cast<unsigned char>(content[pos])) || content[pos] == '-')) {
                ++pos;
            }
            std::string num_str = content.substr(num_start, pos - num_start);
            if (!num_str.empty()) {
                vocab_[key] = std::stoll(num_str);
            }
        }

        init_special_tokens();
        return !vocab_.empty();
    }

    bool is_loaded() const { return !vocab_.empty(); }

    std::vector<int64_t> encode(const std::string& text, size_t max_length = 512) const {
        if (vocab_.empty()) {
            return {cls_token_id_, sep_token_id_};
        }

        std::vector<std::string> words = basic_tokenize(text);
        std::vector<int64_t> ids;
        ids.push_back(cls_token_id_);

        for (const auto& word : words) {
            auto sub_ids = wordpiece_tokenize(word);
            for (int64_t id : sub_ids) {
                if (ids.size() + 1 >= max_length) break; // Reserve 1 for [SEP]
                ids.push_back(id);
            }
            if (ids.size() + 1 >= max_length) break;
        }

        ids.push_back(sep_token_id_);
        return ids;
    }

private:
    std::unordered_map<std::string, int64_t> vocab_;
    int64_t pad_token_id_{0};
    int64_t unk_token_id_{100};
    int64_t cls_token_id_{101};
    int64_t sep_token_id_{102};

    void init_special_tokens() {
        auto it = vocab_.find("[PAD]"); if (it != vocab_.end()) pad_token_id_ = it->second;
        it = vocab_.find("[UNK]"); if (it != vocab_.end()) unk_token_id_ = it->second;
        it = vocab_.find("[CLS]"); if (it != vocab_.end()) cls_token_id_ = it->second;
        it = vocab_.find("[SEP]"); if (it != vocab_.end()) sep_token_id_ = it->second;
    }

    static bool is_punctuation(char c) {
        unsigned char uc = static_cast<unsigned char>(c);
        return (uc >= 33 && uc <= 47) || (uc >= 58 && uc <= 64) ||
               (uc >= 91 && uc <= 96) || (uc >= 123 && uc <= 126);
    }

    std::vector<std::string> basic_tokenize(const std::string& text) const {
        std::vector<std::string> tokens;
        std::string current;

        for (char c : text) {
            unsigned char uc = static_cast<unsigned char>(c);
            if (std::isspace(uc)) {
                if (!current.empty()) {
                    tokens.push_back(current);
                    current.clear();
                }
            } else if (is_punctuation(c)) {
                if (!current.empty()) {
                    tokens.push_back(current);
                    current.clear();
                }
                tokens.push_back(std::string(1, c));
            } else {
                current += static_cast<char>(std::tolower(uc));
            }
        }

        if (!current.empty()) {
            tokens.push_back(current);
        }
        return tokens;
    }

    std::vector<int64_t> wordpiece_tokenize(const std::string& word) const {
        if (word.size() > 100) {
            return {unk_token_id_};
        }

        size_t len = word.size();
        size_t start = 0;
        std::vector<int64_t> sub_ids;

        while (start < len) {
            size_t end = len;
            int64_t cur_id = -1;

            while (start < end) {
                std::string sub = (start == 0) ? word.substr(start, end - start)
                                               : ("##" + word.substr(start, end - start));
                auto it = vocab_.find(sub);
                if (it != vocab_.end()) {
                    cur_id = it->second;
                    break;
                }
                --end;
            }

            if (cur_id == -1) {
                return {unk_token_id_};
            }

            sub_ids.push_back(cur_id);
            start = end;
        }

        return sub_ids;
    }
};

} // namespace vectordb
