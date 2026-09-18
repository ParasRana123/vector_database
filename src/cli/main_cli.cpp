#include "vectordb/engine/database.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <fstream>

using namespace vectordb;

void print_banner() {
    std::cout << "\n============================================================\n";
    std::cout << "          VectorDB C++ v1.0 - High Performance Engine       \n";
    std::cout << "============================================================\n";
    std::cout << "Type 'help' for commands, or 'exit' to quit.\n\n";
}

void print_help() {
    std::cout << "Available Commands:\n";
    std::cout << "  CREATE <name> [dim=384] [metric=COSINE|EUCLIDEAN|DOT] [index=HNSW|FLAT]\n";
    std::cout << "  LIST                                      List all collections\n";
    std::cout << "  USE <collection>                          Set active collection\n";
    std::cout << "  INSERT_TEXT <id> <text...>                Embed & insert document\n";
    std::cout << "  SEARCH_TEXT <query...> [k=5]              Semantic search query\n";
    std::cout << "  INSERT <id> <f1,f2,...> [payload]         Insert raw vector\n";
    std::cout << "  SEARCH <f1,f2,...> [k=5]                  Search raw vector\n";
    std::cout << "  IMPORT <file_path>                        Batch import lines from text file\n";
    std::cout << "  GET <id>                                  Get document by ID\n";
    std::cout << "  DELETE <id>                               Delete document by ID\n";
    std::cout << "  SAVE [dir]                                Save DB checkpoint\n";
    std::cout << "  LOAD [dir]                                Load DB checkpoint\n";
    std::cout << "  STATS                                     Show database metrics\n";
    std::cout << "  CLEAR                                     Clear active collection\n";
    std::cout << "  EXIT / QUIT                               Exit CLI\n\n";
}

static std::vector<float> parse_floats(const std::string& str) {
    std::vector<float> values;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::stof(item));
        }
    }
    return values;
}

int main(int argc, char* argv[]) {
    DatabaseConfig config;
    if (argc > 1) {
        config.data_dir = argv[1];
    }

    Database db(config);
    print_banner();

    // Create a default collection "default" with MiniLM 384 dim
    CollectionConfig def_cfg;
    def_cfg.name = "default";
    def_cfg.dimension = 384;
    def_cfg.metric = DistanceMetric::COSINE;
    def_cfg.index_type = IndexType::HNSW;
    db.create_collection(def_cfg);

    std::string active_collection = "default";
    std::cout << "Active collection: '" << active_collection << "' (384-dim, HNSW, Cosine)\n";

    std::string line;
    while (true) {
        std::cout << "vectordb [" << active_collection << "]> ";
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cmd;
        ss >> cmd;

        std::string cmd_upper = cmd;
        for (char& c : cmd_upper) c = static_cast<char>(std::toupper(c));

        if (cmd_upper == "EXIT" || cmd_upper == "QUIT") {
            std::cout << "Exiting VectorDB. Goodbye!\n";
            break;
        } else if (cmd_upper == "HELP") {
            print_help();
        } else if (cmd_upper == "LIST") {
            auto colls = db.list_collections();
            std::cout << "Collections (" << colls.size() << "):\n";
            for (const auto& name : colls) {
                auto c = db.get_collection(name);
                std::cout << "  - " << name << " [dim=" << c->dimension() 
                          << ", metric=" << distance_metric_to_string(c->metric())
                          << ", index=" << index_type_to_string(c->index_type())
                          << ", docs=" << c->size() << "]\n";
            }
        } else if (cmd_upper == "USE") {
            std::string name;
            ss >> name;
            if (db.has_collection(name)) {
                active_collection = name;
                std::cout << "Switched to collection: " << active_collection << "\n";
            } else {
                std::cout << "Collection '" << name << "' does not exist.\n";
            }
        } else if (cmd_upper == "CREATE") {
            std::string name;
            size_t dim = 384;
            std::string metric_str = "COSINE";
            std::string index_str = "HNSW";

            ss >> name;
            if (name.empty()) {
                std::cout << "Usage: CREATE <name> [dim] [metric] [index]\n";
                continue;
            }
            if (ss >> dim) {
                if (ss >> metric_str) {
                    ss >> index_str;
                }
            }
            try {
                CollectionConfig cfg;
                cfg.name = name;
                cfg.dimension = dim;
                cfg.metric = parse_distance_metric(metric_str);
                cfg.index_type = parse_index_type(index_str);

                if (db.create_collection(cfg)) {
                    active_collection = name;
                    std::cout << "Created collection '" << name << "'\n";
                } else {
                    std::cout << "Collection '" << name << "' already exists.\n";
                }
            } catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << "\n";
            }
        } else if (cmd_upper == "INSERT_TEXT") {
            VectorId id = 0;
            if (!(ss >> id)) {
                std::cout << "Usage: INSERT_TEXT <id> <text...>\n";
                continue;
            }
            std::string text;
            std::getline(ss >> std::ws, text);
            if (text.empty()) {
                std::cout << "Error: Empty text provided.\n";
                continue;
            }
            auto start = std::chrono::high_resolution_clock::now();
            db.insert_text(active_collection, id, text);
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "Inserted doc ID " << id << " (" << elapsed / 1000.0 << " ms)\n";
        } else if (cmd_upper == "SEARCH_TEXT") {
            std::string rest;
            std::getline(ss >> std::ws, rest);
            if (rest.empty()) {
                std::cout << "Usage: SEARCH_TEXT <query...> [k=5]\n";
                continue;
            }

            size_t top_k = 5;
            // Check if last token is a number for top_k
            size_t last_space = rest.rfind(' ');
            std::string query_text = rest;
            if (last_space != std::string::npos) {
                try {
                    size_t parsed_k = std::stoull(rest.substr(last_space + 1));
                    top_k = parsed_k;
                    query_text = rest.substr(0, last_space);
                } catch (...) {}
            }

            auto start = std::chrono::high_resolution_clock::now();
            auto results = db.search_text(active_collection, query_text, top_k);
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start).count();

            std::cout << "\nResults for \"" << query_text << "\" (" << results.size() 
                      << " found in " << std::fixed << std::setprecision(3) << elapsed / 1000.0 << " ms):\n";
            for (size_t i = 0; i < results.size(); ++i) {
                std::cout << "  " << (i + 1) << ". [ID: " << results[i].id 
                          << " | Score: " << std::fixed << std::setprecision(4) << results[i].score << "]\n";
                if (!results[i].payload.empty()) {
                    std::cout << "     \"" << results[i].payload << "\"\n";
                }
            }
            std::cout << "\n";
        } else if (cmd_upper == "INSERT") {
            VectorId id = 0;
            std::string vec_str;
            if (!(ss >> id >> vec_str)) {
                std::cout << "Usage: INSERT <id> <v1,v2,v3...> [payload]\n";
                continue;
            }
            std::string payload;
            std::getline(ss >> std::ws, payload);
            try {
                Vector v = parse_floats(vec_str);
                db.insert_vector(active_collection, id, v, payload);
                std::cout << "Inserted vector ID " << id << "\n";
            } catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << "\n";
            }
        } else if (cmd_upper == "SEARCH") {
            std::string vec_str;
            size_t top_k = 5;
            if (!(ss >> vec_str)) {
                std::cout << "Usage: SEARCH <v1,v2,v3...> [k=5]\n";
                continue;
            }
            ss >> top_k;
            try {
                Vector q = parse_floats(vec_str);
                auto start = std::chrono::high_resolution_clock::now();
                auto results = db.search_vector(active_collection, q, top_k);
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start).count();

                std::cout << "Results (" << results.size() << " in " << elapsed / 1000.0 << " ms):\n";
                for (size_t i = 0; i < results.size(); ++i) {
                    std::cout << "  " << (i + 1) << ". ID: " << results[i].id 
                              << " | Score: " << results[i].score << " | Payload: " << results[i].payload << "\n";
                }
            } catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << "\n";
            }
        } else if (cmd_upper == "IMPORT") {
            std::string filepath;
            ss >> filepath;
            if (filepath.empty()) {
                std::cout << "Usage: IMPORT <file_path>\n";
                continue;
            }
            std::ifstream file(filepath);
            if (!file) {
                std::cout << "Could not open file: " << filepath << "\n";
                continue;
            }
            std::string line_in;
            size_t count = 0;
            auto start = std::chrono::high_resolution_clock::now();
            VectorId next_id = db.get_collection(active_collection)->size() + 1;
            while (std::getline(file, line_in)) {
                if (line_in.empty()) continue;
                db.insert_text(active_collection, next_id++, line_in);
                ++count;
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "Imported " << count << " documents in " << elapsed << " ms\n";
        } else if (cmd_upper == "GET") {
            VectorId id = 0;
            if (ss >> id) {
                auto coll = db.get_collection(active_collection);
                auto doc = coll->get(id);
                if (doc) {
                    std::cout << "Document ID: " << doc->id << "\n";
                    std::cout << "Payload: " << doc->payload << "\n";
                    std::cout << "Metadata: " << doc->metadata.to_json() << "\n";
                } else {
                    std::cout << "Document ID " << id << " not found.\n";
                }
            }
        } else if (cmd_upper == "DELETE") {
            VectorId id = 0;
            if (ss >> id) {
                auto coll = db.get_collection(active_collection);
                if (coll->remove(id)) {
                    std::cout << "Deleted ID " << id << "\n";
                } else {
                    std::cout << "ID " << id << " not found.\n";
                }
            }
        } else if (cmd_upper == "SAVE") {
            std::string dir;
            ss >> dir;
            if (db.save_all(dir)) {
                std::cout << "Database checkpoint saved successfully.\n";
            } else {
                std::cout << "Failed to save checkpoint.\n";
            }
        } else if (cmd_upper == "LOAD") {
            std::string dir;
            ss >> dir;
            if (db.load_all(dir)) {
                std::cout << "Database loaded successfully.\n";
            } else {
                std::cout << "Failed to load database.\n";
            }
        } else if (cmd_upper == "STATS") {
            auto stats = db.get_stats();
            std::cout << "Database Statistics:\n";
            std::cout << "  Collections: " << stats.total_collections << "\n";
            std::cout << "  Total Documents: " << stats.total_documents << "\n";
            std::cout << "  MiniLM Embedder: " << (stats.embedder_ready ? "ONLINE" : "OFFLINE") << "\n";
        } else if (cmd_upper == "CLEAR") {
            db.get_collection(active_collection)->clear();
            std::cout << "Collection '" << active_collection << "' cleared.\n";
        } else {
            std::cout << "Unknown command: '" << cmd << "'. Type 'help' for available commands.\n";
        }
    }

    return 0;
}
