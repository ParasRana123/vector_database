#include "vectordb/engine/database.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
#include <map>

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

class HttpServer {
public:
    HttpServer(Database& db, int port = 8080) : db_(db), port_(port) {}

    void start() {
#if defined(_WIN32)
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "WSAStartup failed\n";
            return;
        }
#endif

        socket_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd == INVALID_SOCKET) {
            std::cerr << "Socket creation failed\n";
            return;
        }

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(static_cast<u_short>(port_));

        if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == SOCKET_ERROR) {
            std::cerr << "Bind failed on port " << port_ << "\n";
            closesocket(server_fd);
            return;
        }

        if (listen(server_fd, 10) == SOCKET_ERROR) {
            std::cerr << "Listen failed\n";
            closesocket(server_fd);
            return;
        }

        std::cout << "[HttpServer] VectorDB REST API Server running at http://127.0.0.1:" << port_ << "\n";
        std::cout << "Endpoints:\n";
        std::cout << "  GET  /api/stats\n";
        std::cout << "  GET  /api/collections\n";
        std::cout << "  POST /api/insert\n";
        std::cout << "  POST /api/search\n\n";

        while (true) {
            sockaddr_in client_addr{};
            socklen_t addrlen = sizeof(client_addr);
            socket_t client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addrlen);
            if (client_fd == INVALID_SOCKET) continue;

            handle_client(client_fd);
        }

#if defined(_WIN32)
        WSACleanup();
#endif
    }

private:
    Database& db_;
    int port_;

    void handle_client(socket_t client_fd) {
        char buffer[8192] = {0};
        int bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
        if (bytes_read <= 0) {
            closesocket(client_fd);
            return;
        }

        std::string request(buffer, bytes_read);
        std::stringstream ss(request);
        std::string method, path, version;
        ss >> method >> path >> version;

        // Parse body if POST
        std::string body;
        size_t header_end = request.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            body = request.substr(header_end + 4);
        }

        std::string response_json = "{}";
        int status_code = 200;

        if (method == "GET" && path == "/api/stats") {
            auto stats = db_.get_stats();
            std::ostringstream out;
            out << "{\"total_collections\":" << stats.total_collections
                << ",\"total_documents\":" << stats.total_documents
                << ",\"embedder_ready\":" << (stats.embedder_ready ? "true" : "false") << "}";
            response_json = out.str();
        } else if (method == "GET" && path == "/api/collections") {
            auto colls = db_.list_collections();
            std::ostringstream out;
            out << "[";
            for (size_t i = 0; i < colls.size(); ++i) {
                if (i > 0) out << ",";
                auto c = db_.get_collection(colls[i]);
                out << "{\"name\":\"" << colls[i] << "\",\"size\":" << c->size()
                    << ",\"dimension\":" << c->dimension()
                    << ",\"metric\":\"" << distance_metric_to_string(c->metric()) << "\"}";
            }
            out << "]";
            response_json = out.str();
        } else if (method == "POST" && path == "/api/insert") {
            Metadata meta = Metadata::from_json(body);
            auto text_opt = meta.get_string("text");
            auto id_opt = meta.get_int("id");
            auto coll_opt = meta.get_string("collection");
            std::string coll_name = coll_opt ? *coll_opt : "default";

            if (text_opt && id_opt) {
                db_.insert_text(coll_name, static_cast<VectorId>(*id_opt), *text_opt);
                response_json = "{\"status\":\"ok\",\"inserted_id\":" + std::to_string(*id_opt) + "}";
            } else {
                status_code = 400;
                response_json = "{\"error\":\"Missing 'id' or 'text'\"}";
            }
        } else if (method == "POST" && path == "/api/search") {
            Metadata meta = Metadata::from_json(body);
            auto query_opt = meta.get_string("query");
            auto top_k_opt = meta.get_int("top_k");
            auto coll_opt = meta.get_string("collection");
            std::string coll_name = coll_opt ? *coll_opt : "default";
            size_t top_k = top_k_opt ? static_cast<size_t>(*top_k_opt) : 5;

            if (query_opt) {
                auto start = std::chrono::high_resolution_clock::now();
                auto results = db_.search_text(coll_name, *query_opt, top_k);
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start).count();

                std::ostringstream out;
                out << "{\"query\":\"" << *query_opt << "\",\"latency_ms\":" << (elapsed / 1000.0)
                    << ",\"results\":[";
                for (size_t i = 0; i < results.size(); ++i) {
                    if (i > 0) out << ",";
                    out << "{\"id\":" << results[i].id
                        << ",\"score\":" << results[i].score
                        << ",\"payload\":\"" << results[i].payload << "\"}";
                }
                out << "]}";
                response_json = out.str();
            } else {
                status_code = 400;
                response_json = "{\"error\":\"Missing 'query' parameter\"}";
            }
        } else {
            status_code = 404;
            response_json = "{\"error\":\"Endpoint not found\"}";
        }

        std::ostringstream response;
        response << "HTTP/1.1 " << status_code << " OK\r\n"
                 << "Content-Type: application/json\r\n"
                 << "Access-Control-Allow-Origin: *\r\n"
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
        port = std::stoi(argv[1]);
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
