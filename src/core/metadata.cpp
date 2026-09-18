#include "vectordb/core/metadata.hpp"
#include <sstream>
#include <iomanip>
#include <cctype>

namespace vectordb {

void Metadata::set(const std::string& key, const MetadataValue& value) {
    fields_[key] = value;
}

void Metadata::set_string(const std::string& key, const std::string& value) {
    fields_[key] = value;
}

void Metadata::set_int(const std::string& key, int64_t value) {
    fields_[key] = value;
}

void Metadata::set_double(const std::string& key, double value) {
    fields_[key] = value;
}

void Metadata::set_bool(const std::string& key, bool value) {
    fields_[key] = value;
}

bool Metadata::has(const std::string& key) const {
    return fields_.find(key) != fields_.end();
}

std::optional<MetadataValue> Metadata::get(const std::string& key) const {
    auto it = fields_.find(key);
    if (it != fields_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<std::string> Metadata::get_string(const std::string& key) const {
    auto val = get(key);
    if (val && std::holds_alternative<std::string>(*val)) {
        return std::get<std::string>(*val);
    }
    return std::nullopt;
}

std::optional<int64_t> Metadata::get_int(const std::string& key) const {
    auto val = get(key);
    if (val) {
        if (std::holds_alternative<int64_t>(*val)) return std::get<int64_t>(*val);
        if (std::holds_alternative<double>(*val)) return static_cast<int64_t>(std::get<double>(*val));
    }
    return std::nullopt;
}

std::optional<double> Metadata::get_double(const std::string& key) const {
    auto val = get(key);
    if (val) {
        if (std::holds_alternative<double>(*val)) return std::get<double>(*val);
        if (std::holds_alternative<int64_t>(*val)) return static_cast<double>(std::get<int64_t>(*val));
    }
    return std::nullopt;
}

std::optional<bool> Metadata::get_bool(const std::string& key) const {
    auto val = get(key);
    if (val && std::holds_alternative<bool>(*val)) {
        return std::get<bool>(*val);
    }
    return std::nullopt;
}

static std::string escape_json_string(const std::string& str) {
    std::string out = "\"";
    for (char c : str) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\b') out += "\\b";
        else if (c == '\f') out += "\\f";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    out += "\"";
    return out;
}

std::string Metadata::to_json() const {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto& [k, v] : fields_) {
        if (!first) oss << ",";
        first = false;
        oss << escape_json_string(k) << ":";
        std::visit([&oss](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::string>) {
                oss << escape_json_string(arg);
            } else if constexpr (std::is_same_v<T, int64_t>) {
                oss << arg;
            } else if constexpr (std::is_same_v<T, double>) {
                oss << std::fixed << std::setprecision(6) << arg;
            } else if constexpr (std::is_same_v<T, bool>) {
                oss << (arg ? "true" : "false");
            }
        }, v);
    }
    oss << "}";
    return oss.str();
}

static void skip_ws(const std::string& s, size_t& pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
}

static std::string parse_string_token(const std::string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != '"') return "";
    ++pos; // skip quote
    std::string res;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            ++pos;
            if (s[pos] == 'n') res += '\n';
            else if (s[pos] == 't') res += '\t';
            else if (s[pos] == 'r') res += '\r';
            else res += s[pos];
        } else {
            res += s[pos];
        }
        ++pos;
    }
    if (pos < s.size() && s[pos] == '"') ++pos;
    return res;
}

Metadata Metadata::from_json(const std::string& json_str) {
    Metadata meta;
    size_t pos = 0;
    skip_ws(json_str, pos);
    if (pos >= json_str.size() || json_str[pos] != '{') return meta;
    ++pos;

    while (pos < json_str.size()) {
        skip_ws(json_str, pos);
        if (pos >= json_str.size() || json_str[pos] == '}') break;

        std::string key = parse_string_token(json_str, pos);
        skip_ws(json_str, pos);
        if (pos < json_str.size() && json_str[pos] == ':') ++pos;
        skip_ws(json_str, pos);

        if (pos >= json_str.size()) break;

        if (json_str[pos] == '"') {
            std::string s_val = parse_string_token(json_str, pos);
            meta.set_string(key, s_val);
        } else if (json_str.substr(pos, 4) == "true") {
            meta.set_bool(key, true);
            pos += 4;
        } else if (json_str.substr(pos, 5) == "false") {
            meta.set_bool(key, false);
            pos += 5;
        } else {
            // numeric
            size_t start = pos;
            bool is_double = false;
            while (pos < json_str.size() && (std::isdigit(static_cast<unsigned char>(json_str[pos])) ||
                   json_str[pos] == '-' || json_str[pos] == '+' || json_str[pos] == '.' || json_str[pos] == 'e' || json_str[pos] == 'E')) {
                if (json_str[pos] == '.' || json_str[pos] == 'e' || json_str[pos] == 'E') is_double = true;
                ++pos;
            }
            std::string num_str = json_str.substr(start, pos - start);
            if (!num_str.empty()) {
                if (is_double) {
                    meta.set_double(key, std::stod(num_str));
                } else {
                    meta.set_int(key, std::stoll(num_str));
                }
            }
        }

        skip_ws(json_str, pos);
        if (pos < json_str.size() && json_str[pos] == ',') ++pos;
    }
    return meta;
}

static bool compare_values(const MetadataValue& actual, FilterOp op, const MetadataValue& target) {
    if (actual.index() == target.index()) {
        if (std::holds_alternative<std::string>(actual)) {
            const auto& a = std::get<std::string>(actual);
            const auto& b = std::get<std::string>(target);
            if (op == FilterOp::EQUALS) return a == b;
            if (op == FilterOp::NOT_EQUALS) return a != b;
            if (op == FilterOp::GREATER_THAN) return a > b;
            if (op == FilterOp::GREATER_THAN_OR_EQUAL) return a >= b;
            if (op == FilterOp::LESS_THAN) return a < b;
            if (op == FilterOp::LESS_THAN_OR_EQUAL) return a <= b;
        } else if (std::holds_alternative<bool>(actual)) {
            bool a = std::get<bool>(actual);
            bool b = std::get<bool>(target);
            if (op == FilterOp::EQUALS) return a == b;
            if (op == FilterOp::NOT_EQUALS) return a != b;
        }
    }

    // Number comparison (handles int64 vs double)
    double a_num = 0.0, b_num = 0.0;
    bool a_is_num = false, b_is_num = false;

    if (std::holds_alternative<int64_t>(actual)) { a_num = static_cast<double>(std::get<int64_t>(actual)); a_is_num = true; }
    else if (std::holds_alternative<double>(actual)) { a_num = std::get<double>(actual); a_is_num = true; }

    if (std::holds_alternative<int64_t>(target)) { b_num = static_cast<double>(std::get<int64_t>(target)); b_is_num = true; }
    else if (std::holds_alternative<double>(target)) { b_num = std::get<double>(target); b_is_num = true; }

    if (a_is_num && b_is_num) {
        if (op == FilterOp::EQUALS) return std::abs(a_num - b_num) < 1e-9;
        if (op == FilterOp::NOT_EQUALS) return std::abs(a_num - b_num) >= 1e-9;
        if (op == FilterOp::GREATER_THAN) return a_num > b_num;
        if (op == FilterOp::GREATER_THAN_OR_EQUAL) return a_num >= b_num;
        if (op == FilterOp::LESS_THAN) return a_num < b_num;
        if (op == FilterOp::LESS_THAN_OR_EQUAL) return a_num <= b_num;
    }

    if (op == FilterOp::NOT_EQUALS) return true;
    return false;
}

bool FieldFilter::matches(const Metadata& metadata) const {
    auto val = metadata.get(field_);
    if (!val.has_value()) {
        return op_ == FilterOp::NOT_EQUALS;
    }

    if (op_ == FilterOp::IN) {
        for (const auto& in_v : in_values_) {
            if (compare_values(*val, FilterOp::EQUALS, in_v)) {
                return true;
            }
        }
        return false;
    }

    return compare_values(*val, op_, value_);
}

bool CompositeFilter::matches(const Metadata& metadata) const {
    if (op_ == FilterOp::AND) {
        for (const auto& child : children_) {
            if (!child || !child->matches(metadata)) return false;
        }
        return true;
    } else if (op_ == FilterOp::OR) {
        for (const auto& child : children_) {
            if (child && child->matches(metadata)) return true;
        }
        return false;
    } else if (op_ == FilterOp::NOT) {
        if (!children_.empty() && children_[0]) {
            return !children_[0]->matches(metadata);
        }
        return true;
    }
    return true;
}

namespace filter {
    std::shared_ptr<Filter> eq(const std::string& field, const MetadataValue& val) {
        return std::make_shared<FieldFilter>(field, FilterOp::EQUALS, val);
    }
    std::shared_ptr<Filter> eq(const std::string& field, const char* val) {
        return std::make_shared<FieldFilter>(field, FilterOp::EQUALS, MetadataValue(std::string(val)));
    }
    std::shared_ptr<Filter> eq(const std::string& field, const std::string& val) {
        return std::make_shared<FieldFilter>(field, FilterOp::EQUALS, MetadataValue(val));
    }
    std::shared_ptr<Filter> eq(const std::string& field, int val) {
        return std::make_shared<FieldFilter>(field, FilterOp::EQUALS, MetadataValue(int64_t(val)));
    }

    std::shared_ptr<Filter> ne(const std::string& field, const MetadataValue& val) {
        return std::make_shared<FieldFilter>(field, FilterOp::NOT_EQUALS, val);
    }
    std::shared_ptr<Filter> ne(const std::string& field, const char* val) {
        return std::make_shared<FieldFilter>(field, FilterOp::NOT_EQUALS, MetadataValue(std::string(val)));
    }
    std::shared_ptr<Filter> ne(const std::string& field, const std::string& val) {
        return std::make_shared<FieldFilter>(field, FilterOp::NOT_EQUALS, MetadataValue(val));
    }

    std::shared_ptr<Filter> gt(const std::string& field, const MetadataValue& val) {
        return std::make_shared<FieldFilter>(field, FilterOp::GREATER_THAN, val);
    }
    std::shared_ptr<Filter> gt(const std::string& field, int val) {
        return std::make_shared<FieldFilter>(field, FilterOp::GREATER_THAN, MetadataValue(int64_t(val)));
    }

    std::shared_ptr<Filter> gte(const std::string& field, const MetadataValue& val) {
        return std::make_shared<FieldFilter>(field, FilterOp::GREATER_THAN_OR_EQUAL, val);
    }
    std::shared_ptr<Filter> gte(const std::string& field, int val) {
        return std::make_shared<FieldFilter>(field, FilterOp::GREATER_THAN_OR_EQUAL, MetadataValue(int64_t(val)));
    }

    std::shared_ptr<Filter> lt(const std::string& field, const MetadataValue& val) {
        return std::make_shared<FieldFilter>(field, FilterOp::LESS_THAN, val);
    }
    std::shared_ptr<Filter> lt(const std::string& field, int val) {
        return std::make_shared<FieldFilter>(field, FilterOp::LESS_THAN, MetadataValue(int64_t(val)));
    }

    std::shared_ptr<Filter> lte(const std::string& field, const MetadataValue& val) {
        return std::make_shared<FieldFilter>(field, FilterOp::LESS_THAN_OR_EQUAL, val);
    }
    std::shared_ptr<Filter> lte(const std::string& field, int val) {
        return std::make_shared<FieldFilter>(field, FilterOp::LESS_THAN_OR_EQUAL, MetadataValue(int64_t(val)));
    }

    std::shared_ptr<Filter> in(const std::string& field, const std::vector<MetadataValue>& vals) {
        return std::make_shared<FieldFilter>(field, FilterOp::IN, MetadataValue(""), vals);
    }
    std::shared_ptr<Filter> in(const std::string& field, const std::vector<std::string>& vals) {
        std::vector<MetadataValue> meta_vals;
        meta_vals.reserve(vals.size());
        for (const auto& s : vals) meta_vals.emplace_back(s);
        return std::make_shared<FieldFilter>(field, FilterOp::IN, MetadataValue(""), meta_vals);
    }

    std::shared_ptr<Filter> all_of(const std::vector<std::shared_ptr<Filter>>& filters) {
        return std::make_shared<CompositeFilter>(FilterOp::AND, filters);
    }
    std::shared_ptr<Filter> any_of(const std::vector<std::shared_ptr<Filter>>& filters) {
        return std::make_shared<CompositeFilter>(FilterOp::OR, filters);
    }
    std::shared_ptr<Filter> not_filter(const std::shared_ptr<Filter>& f) {
        return std::make_shared<CompositeFilter>(FilterOp::NOT, std::vector<std::shared_ptr<Filter>>{f});
    }
}

} // namespace vectordb
