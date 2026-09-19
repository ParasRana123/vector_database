#include "vectordb/engine/database.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
#include <map>
#include <algorithm>
#include <cctype>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
using socklen_t = int;
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
using socket_t = int;
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define closesocket close
#endif

using namespace vectordb;

static std::string url_decode(const std::string& str) {
    std::string result;
    result.reserve(str.size());
    for (size_t i = 0; i < str.size(); ++i) {
        if (str[i] == '+') {
            result += ' ';
        } else if (str[i] == '%' && i + 2 < str.size() && std::isxdigit(static_cast<unsigned char>(str[i+1])) && std::isxdigit(static_cast<unsigned char>(str[i+2]))) {
            int val = 0;
            std::istringstream iss(str.substr(i + 1, 2));
            if (iss >> std::hex >> val) {
                result += static_cast<char>(val);
                i += 2;
            } else {
                result += '%';
            }
        } else {
            result += str[i];
        }
    }
    return result;
}

static std::map<std::string, std::string> parse_query_params(const std::string& query_str) {
    std::map<std::string, std::string> params;
    std::istringstream iss(query_str);
    std::string pair;
    while (std::getline(iss, pair, '&')) {
        if (pair.empty()) continue;
        size_t eq = pair.find('=');
        if (eq != std::string::npos) {
            std::string key = url_decode(pair.substr(0, eq));
            std::string val = url_decode(pair.substr(eq + 1));
            params[key] = val;
        } else {
            params[url_decode(pair)] = "";
        }
    }
    return params;
}

class HttpServer {
public:
    HttpServer(Database& db, int port = 8080) : db_(db), port_(port) {}

    void start() {
#if defined(_WIN32)
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "[HttpServer] WSAStartup failed\n";
            return;
        }
#endif

        socket_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd == INVALID_SOCKET) {
            std::cerr << "[HttpServer] Socket creation failed\n";
            return;
        }

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(static_cast<u_short>(port_));

        if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == SOCKET_ERROR) {
            std::cerr << "[HttpServer] Bind failed on port " << port_ << "\n";
            closesocket(server_fd);
            return;
        }

        if (listen(server_fd, 64) == SOCKET_ERROR) {
            std::cerr << "[HttpServer] Listen failed\n";
            closesocket(server_fd);
            return;
        }

        std::cout << "[HttpServer] VectorDB REST API Server running at http://0.0.0.0:" << port_ << "\n";
        std::cout << "Endpoints:\n";
        std::cout << "  GET  /health\n";
        std::cout << "  GET  /api/stats\n";
        std::cout << "  GET  /api/collections\n";
        std::cout << "  POST /api/insert\n";
        std::cout << "  POST /api/search\n\n";

        while (true) {
            sockaddr_in client_addr{};
            socklen_t addrlen = sizeof(client_addr);
            socket_t client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addrlen);
            if (client_fd == INVALID_SOCKET) continue;

            // Handle each client request in a separate thread
            std::thread([this, client_fd]() {
                this->handle_client(client_fd);
            }).detach();
        }

#if defined(_WIN32)
        WSACleanup();
#endif
    }

private:
    Database& db_;
    int port_;

    void handle_client(socket_t client_fd) {
        // Set 10-second timeout on receive
#if defined(_WIN32)
        DWORD timeout_ms = 10000;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
#else
        struct timeval tv{};
        tv.tv_sec = 10;
        tv.tv_usec = 0;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
#endif

        std::string raw_request;
        raw_request.reserve(4096);
        char buffer[4096];
        size_t header_end = std::string::npos;
        size_t header_sep_len = 0;

        // 1. Read header until \r\n\r\n or \n\n
        while (raw_request.size() < 65536) {
            header_end = raw_request.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                header_sep_len = 4;
                break;
            }
            header_end = raw_request.find("\n\n");
            if (header_end != std::string::npos) {
                header_sep_len = 2;
                break;
            }

            int bytes_read = recv(client_fd, buffer, sizeof(buffer), 0);
            if (bytes_read <= 0) break;
            raw_request.append(buffer, bytes_read);
        }

        if (header_end == std::string::npos) {
            closesocket(client_fd);
            return;
        }

        std::string headers_part = raw_request.substr(0, header_end);
        std::string body = raw_request.substr(header_end + header_sep_len);

        std::stringstream ss(headers_part);
        std::string method, full_path, version;
        ss >> method >> full_path >> version;

        // Parse Content-Length
        size_t content_length = 0;
        {
            std::string lower_headers = headers_part;
            std::transform(lower_headers.begin(), lower_headers.end(), lower_headers.begin(), ::tolower);
            std::string cl_tag = "content-length:";
            size_t cl_pos = lower_headers.find(cl_tag);
            if (cl_pos != std::string::npos) {
                size_t val_start = cl_pos + cl_tag.size();
                size_t val_end = lower_headers.find("\n", val_start);
                std::string val_str = lower_headers.substr(val_start, val_end == std::string::npos ? std::string::npos : val_end - val_start);
                try {
                    content_length = std::stoull(val_str);
                } catch (...) {
                    content_length = 0;
                }
            }
        }

        // 2. Read full body if content_length > current body size
        while (body.size() < content_length) {
            size_t needed = content_length - body.size();
            size_t to_read = (needed < sizeof(buffer)) ? needed : sizeof(buffer);
            int bytes_read = recv(client_fd, buffer, static_cast<int>(to_read), 0);
            if (bytes_read <= 0) break;
            body.append(buffer, bytes_read);
        }

        // Extract clean path and query string
        std::string clean_path = full_path;
        std::string query_string;
        size_t qpos = full_path.find('?');
        if (qpos != std::string::npos) {
            clean_path = full_path.substr(0, qpos);
            query_string = full_path.substr(qpos + 1);
        }
        while (clean_path.size() > 1 && clean_path.back() == '/') {
            clean_path.pop_back();
        }

        auto url_params = parse_query_params(query_string);

        // Handle CORS Preflight (OPTIONS)
        if (method == "OPTIONS") {
            std::ostringstream response;
            response << "HTTP/1.1 200 OK\r\n"
                     << "Access-Control-Allow-Origin: *\r\n"
                     << "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
                     << "Access-Control-Allow-Headers: Content-Type, Authorization, Accept, X-Requested-With\r\n"
                     << "Content-Length: 0\r\n"
                     << "Connection: close\r\n\r\n";
            std::string resp_str = response.str();
            send(client_fd, resp_str.data(), static_cast<int>(resp_str.size()), 0);
            closesocket(client_fd);
            return;
        }

        std::string response_json = "{}";
        int status_code = 200;

        if (method == "GET" && (clean_path.empty() || clean_path == "/" || clean_path == "/health" || clean_path == "/status" || clean_path == "/api" || clean_path == "/api/health")) {
            auto stats = db_.get_stats();
            std::ostringstream out;
            out << "{"
                << "\"status\":\"online\","
                << "\"engine\":\"VectorDB C++ v1.0\","
                << "\"message\":\"VectorDB REST API is operational and healthy\","
                << "\"stats\":{"
                << "\"total_collections\":" << stats.total_collections << ","
                << "\"total_documents\":" << stats.total_documents << ","
                << "\"embedder_ready\":" << (stats.embedder_ready ? "true" : "false")
                << "},"
                << "\"endpoints\":{"
                << "\"GET /\":\"Server status & API documentation\","
                << "\"GET /health\":\"Service health check\","
                << "\"GET /api/stats\":\"Database metrics & document counts\","
                << "\"GET /api/collections\":\"List all collections & schemas\","
                << "\"POST /api/insert\":\"Insert document {\\\"collection\\\":\\\"default\\\",\\\"id\\\":1,\\\"text\\\":\\\"...\\\"}\","
                << "\"POST /api/search\":\"Semantic search {\\\"collection\\\":\\\"default\\\",\\\"query\\\":\\\"...\\\",\\\"top_k\\\":5}\""
                << "}"
                << "}";
            response_json = out.str();
        } else if (method == "GET" && (clean_path == "/api/stats" || clean_path == "/stats")) {
            auto stats = db_.get_stats();
            std::ostringstream out;
            out << "{\"total_collections\":" << stats.total_collections
                << ",\"total_documents\":" << stats.total_documents
                << ",\"embedder_ready\":" << (stats.embedder_ready ? "true" : "false") << "}";
            response_json = out.str();
        } else if (method == "GET" && (clean_path == "/api/collections" || clean_path == "/collections")) {
            auto colls = db_.list_collections();
            std::ostringstream out;
            out << "[";
            for (size_t i = 0; i < colls.size(); ++i) {
                if (i > 0) out << ",";
                auto c = db_.get_collection(colls[i]);
                out << "{\"name\":\"" << colls[i] << "\",\"size\":" << (c ? c->size() : 0)
                    << ",\"dimension\":" << (c ? c->dimension() : 0)
                    << ",\"metric\":\"" << (c ? distance_metric_to_string(c->metric()) : "COSINE") << "\"}";
            }
            out << "]";
            response_json = out.str();
        } else if ((method == "POST" || method == "GET") && (clean_path == "/api/insert" || clean_path == "/insert")) {
            std::string text_val;
            int64_t id_val = 0;
            std::string coll_name = "default";

            if (!body.empty()) {
                Metadata meta = Metadata::from_json(body);
                if (auto t = meta.get_string("text")) text_val = *t;
                else if (auto d = meta.get_string("document")) text_val = *d;
                else if (auto doc = meta.get_string("doc")) text_val = *doc;
                else if (auto p = meta.get_string("payload")) text_val = *p;
                else if (auto c = meta.get_string("content")) text_val = *c;

                if (auto id_opt = meta.get_int("id")) id_val = *id_opt;
                if (auto coll_opt = meta.get_string("collection")) coll_name = *coll_opt;

                // If not in JSON, check form urlencoded
                if (text_val.empty()) {
                    auto form_params = parse_query_params(body);
                    if (form_params.count("text")) text_val = form_params["text"];
                    else if (form_params.count("document")) text_val = form_params["document"];
                    else if (form_params.count("payload")) text_val = form_params["payload"];

                    if (form_params.count("id")) {
                        try { id_val = std::stoll(form_params["id"]); } catch (...) {}
                    }
                    if (form_params.count("collection")) coll_name = form_params["collection"];
                }
            }

            // Fallback to URL query params
            if (text_val.empty() && url_params.count("text")) text_val = url_params["text"];
            if (id_val == 0 && url_params.count("id")) {
                try { id_val = std::stoll(url_params["id"]); } catch (...) {}
            }
            if (coll_name == "default" && url_params.count("collection")) coll_name = url_params["collection"];

            if (!text_val.empty()) {
                if (id_val == 0) {
                    auto coll = db_.get_collection(coll_name);
                    id_val = coll ? static_cast<int64_t>(coll->size() + 1) : 1;
                }
                try {
                    db_.insert_text(coll_name, static_cast<VectorId>(id_val), text_val);
                    response_json = "{\"status\":\"ok\",\"inserted_id\":" + std::to_string(id_val) + ",\"collection\":\"" + coll_name + "\"}";
                } catch (const std::exception& ex) {
                    status_code = 500;
                    response_json = "{\"error\":\"" + std::string(ex.what()) + "\"}";
                }
            } else {
                status_code = 400;
                response_json = "{\"error\":\"Missing 'text' parameter in insert request\"}";
            }
        } else if ((method == "POST" || method == "GET") && (clean_path == "/api/search" || clean_path == "/search")) {
            std::string query_text;
            size_t top_k = 5;
            std::string coll_name = "default";

            // 1. Try parsing JSON body if body is present
            if (!body.empty()) {
                Metadata meta = Metadata::from_json(body);
                if (auto q = meta.get_string("query")) query_text = *q;
                else if (auto q_short = meta.get_string("q")) query_text = *q_short;
                else if (auto t = meta.get_string("text")) query_text = *t;
                else if (auto qt = meta.get_string("query_text")) query_text = *qt;

                if (auto top_k_opt = meta.get_int("top_k")) top_k = static_cast<size_t>(*top_k_opt);
                else if (auto k_opt = meta.get_int("k")) top_k = static_cast<size_t>(*k_opt);
                else if (auto lim_opt = meta.get_int("limit")) top_k = static_cast<size_t>(*lim_opt);

                if (auto coll_opt = meta.get_string("collection")) coll_name = *coll_opt;

                // 2. If JSON did not match, try parsing body as form urlencoded
                if (query_text.empty()) {
                    auto form_params = parse_query_params(body);
                    if (form_params.count("query")) query_text = form_params["query"];
                    else if (form_params.count("q")) query_text = form_params["q"];
                    else if (form_params.count("text")) query_text = form_params["text"];

                    if (form_params.count("top_k")) {
                        try { top_k = std::stoul(form_params["top_k"]); } catch (...) {}
                    }
                    if (form_params.count("collection")) coll_name = form_params["collection"];
                }

                // 3. If still empty and body doesn't look like JSON/Form, treat raw body as query
                if (query_text.empty() && body.front() != '{' && body.front() != '[') {
                    std::string trimmed = body;
                    size_t first = trimmed.find_first_not_of(" \t\r\n");
                    size_t last = trimmed.find_last_not_of(" \t\r\n");
                    if (first != std::string::npos && last != std::string::npos) {
                        query_text = trimmed.substr(first, (last - first + 1));
                    }
                }
            }

            // 4. Fallback to URL query string parameters (?query=... or ?q=...)
            if (query_text.empty()) {
                if (url_params.count("query")) query_text = url_params["query"];
                else if (url_params.count("q")) query_text = url_params["q"];
                else if (url_params.count("text")) query_text = url_params["text"];
            }

            if (url_params.count("top_k")) {
                try { top_k = std::stoul(url_params["top_k"]); } catch (...) {}
            } else if (url_params.count("k")) {
                try { top_k = std::stoul(url_params["k"]); } catch (...) {}
            } else if (url_params.count("limit")) {
                try { top_k = std::stoul(url_params["limit"]); } catch (...) {}
            }

            if (coll_name == "default" && url_params.count("collection")) {
                coll_name = url_params["collection"];
            }

            if (!query_text.empty()) {
                try {
                    auto start = std::chrono::high_resolution_clock::now();
                    auto results = db_.search_text(coll_name, query_text, top_k);
                    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now() - start).count();

                    std::ostringstream out;
                    out << "{\"query\":\"" << query_text << "\",\"latency_ms\":" << (elapsed / 1000.0)
                        << ",\"collection\":\"" << coll_name << "\""
                        << ",\"results\":[";
                    for (size_t i = 0; i < results.size(); ++i) {
                        if (i > 0) out << ",";
                        // Escape quotes in payload string
                        std::string safe_payload;
                        for (char c : results[i].payload) {
                            if (c == '"') safe_payload += "\\\"";
                            else if (c == '\\') safe_payload += "\\\\";
                            else if (c == '\n') safe_payload += "\\n";
                            else if (c == '\r') safe_payload += "\\r";
                            else if (c == '\t') safe_payload += "\\t";
                            else safe_payload += c;
                        }
                        out << "{\"id\":" << results[i].id
                            << ",\"score\":" << results[i].score
                            << ",\"payload\":\"" << safe_payload << "\"}";
                    }
                    out << "]}";
                    response_json = out.str();
                } catch (const std::exception& ex) {
                    status_code = 500;
                    response_json = "{\"error\":\"" + std::string(ex.what()) + "\"}";
                }
            } else {
                status_code = 400;
                response_json = "{\"error\":\"Missing 'query' parameter\",\"hint\":\"Pass 'query' in JSON body or as query parameter (?query=...)\"}";
            }
        } else {
            status_code = 404;
            response_json = "{\"error\":\"Endpoint not found\",\"path\":\"" + full_path + "\"}";
        }

        std::ostringstream response;
        response << "HTTP/1.1 " << status_code << " OK\r\n"
                 << "Content-Type: application/json\r\n"
                 << "Access-Control-Allow-Origin: *\r\n"
                 << "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
                 << "Access-Control-Allow-Headers: Content-Type, Authorization, Accept, X-Requested-With\r\n"
                 << "Content-Length: " << response_json.size() << "\r\n"
                 << "Connection: close\r\n\r\n"
                 << response_json;

        std::string resp_str = response.str();
        send(client_fd, resp_str.data(), static_cast<int>(resp_str.size()), 0);
        closesocket(client_fd);
    }
};

int main(int argc, char* argv[]) {
    int port = 8080;
    if (argc > 1) {
        try {
            port = std::stoi(argv[1]);
        } catch (...) {
            port = 8080;
        }
    }

    DatabaseConfig config;
    Database db(config);

    CollectionConfig def_cfg;
    def_cfg.name = "default";
    def_cfg.dimension = 384;
    def_cfg.metric = DistanceMetric::COSINE;
    def_cfg.index_type = IndexType::HNSW;
    db.create_collection(def_cfg);

    HttpServer server(db, port);
    server.start();

    return 0;
}
