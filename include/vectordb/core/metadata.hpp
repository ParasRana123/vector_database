#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <memory>
#include <optional>
#include <sstream>

namespace vectordb {

using MetadataValue = std::variant<std::string, int64_t, double, bool>;

class Metadata {
public:
    Metadata() = default;

    void set(const std::string& key, const MetadataValue& value);
    void set_string(const std::string& key, const std::string& value);
    void set_int(const std::string& key, int64_t value);
    void set_double(const std::string& key, double value);
    void set_bool(const std::string& key, bool value);

    bool has(const std::string& key) const;
    std::optional<MetadataValue> get(const std::string& key) const;

    std::optional<std::string> get_string(const std::string& key) const;
    std::optional<int64_t> get_int(const std::string& key) const;
    std::optional<double> get_double(const std::string& key) const;
    std::optional<bool> get_bool(const std::string& key) const;

    const std::unordered_map<std::string, MetadataValue>& data() const { return fields_; }

    std::string to_json() const;
    static Metadata from_json(const std::string& json_str);

private:
    std::unordered_map<std::string, MetadataValue> fields_;
};

enum class FilterOp {
    EQUALS,
    NOT_EQUALS,
    GREATER_THAN,
    GREATER_THAN_OR_EQUAL,
    LESS_THAN,
    LESS_THAN_OR_EQUAL,
    IN,
    AND,
    OR,
    NOT
};

class Filter {
public:
    virtual ~Filter() = default;
    virtual bool matches(const Metadata& metadata) const = 0;
};

class FieldFilter : public Filter {
public:
    FieldFilter(std::string field, FilterOp op, MetadataValue val, std::vector<MetadataValue> in_vals = {})
        : field_(std::move(field)), op_(op), value_(std::move(val)), in_values_(std::move(in_vals)) {}

    bool matches(const Metadata& metadata) const override;

private:
    std::string field_;
    FilterOp op_;
    MetadataValue value_;
    std::vector<MetadataValue> in_values_;
};

class CompositeFilter : public Filter {
public:
    CompositeFilter(FilterOp op, std::vector<std::shared_ptr<Filter>> children)
        : op_(op), children_(std::move(children)) {}

    bool matches(const Metadata& metadata) const override;

private:
    FilterOp op_;
    std::vector<std::shared_ptr<Filter>> children_;
};

// Convenience factory helpers
namespace filter {
    std::shared_ptr<Filter> eq(const std::string& field, const MetadataValue& val);
    std::shared_ptr<Filter> eq(const std::string& field, const char* val);
    std::shared_ptr<Filter> eq(const std::string& field, const std::string& val);
    std::shared_ptr<Filter> eq(const std::string& field, int val);

    std::shared_ptr<Filter> ne(const std::string& field, const MetadataValue& val);
    std::shared_ptr<Filter> ne(const std::string& field, const char* val);
    std::shared_ptr<Filter> ne(const std::string& field, const std::string& val);

    std::shared_ptr<Filter> gt(const std::string& field, const MetadataValue& val);
    std::shared_ptr<Filter> gt(const std::string& field, int val);

    std::shared_ptr<Filter> gte(const std::string& field, const MetadataValue& val);
    std::shared_ptr<Filter> gte(const std::string& field, int val);

    std::shared_ptr<Filter> lt(const std::string& field, const MetadataValue& val);
    std::shared_ptr<Filter> lt(const std::string& field, int val);

    std::shared_ptr<Filter> lte(const std::string& field, const MetadataValue& val);
    std::shared_ptr<Filter> lte(const std::string& field, int val);

    std::shared_ptr<Filter> in(const std::string& field, const std::vector<MetadataValue>& vals);
    std::shared_ptr<Filter> in(const std::string& field, const std::vector<std::string>& vals);

    std::shared_ptr<Filter> all_of(const std::vector<std::shared_ptr<Filter>>& filters);
    std::shared_ptr<Filter> any_of(const std::vector<std::shared_ptr<Filter>>& filters);
    std::shared_ptr<Filter> not_filter(const std::shared_ptr<Filter>& f);
}

} // namespace vectordb
