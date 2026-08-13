#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>
#include <cmath>
#include <random>
#include <functional>
#include <iomanip>
#include <algorithm>
using namespace std;

// Dimensionality of the output embedding vector.
static const int EMBEDDING_DIM = 64;

// =======================================================================
//  EMBEDDING LOGIC  (same approach as your original code)
// =======================================================================

vector<string> tokenize(const string &text) {
    vector<string> tokens;
    string current;
    for (unsigned char ch : text) {
        if (isalnum(ch)) {
            current += static_cast<char>(tolower(ch));
        } else {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

vector<double> word_to_vector(const string &word, int dim) {
    hash<string> hasher;
    size_t seed = hasher(word);
    mt19937_64 rng(seed);
    normal_distribution<double> dist(0.0, 1.0);
    vector<double> vec(dim);
    for (int i = 0; i < dim; ++i) vec[i] = dist(rng);
    return vec;
}

void l2_normalize(vector<double> &vec) {
    double norm = 0.0;
    for (double v : vec) norm += v * v;
    norm = sqrt(norm);
    if (norm > 1e-12) for (double &v : vec) v /= norm;
}

vector<double> embed_query(const string &query, int dim = EMBEDDING_DIM) {
    vector<string> tokens = tokenize(query);
    vector<double> embedding(dim, 0.0);
    if (tokens.empty()) return embedding;
    for (const auto &token : tokens) {
        vector<double> wv = word_to_vector(token, dim);
        for (int i = 0; i < dim; ++i) embedding[i] += wv[i];
    }
    for (double &v : embedding) v /= static_cast<double>(tokens.size());
    l2_normalize(embedding);
    return embedding;
}

double cosine_similarity(const vector<double> &a, const vector<double> &b) {
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot; // both vectors are L2-normalized, so dot == cosine similarity
}

// =======================================================================
//  SMALL STRING HELPERS (to persist multi-line text safely on one line)
// =======================================================================

string escape_text(const string &s) {
    string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

string unescape_text(const string &s) {
    string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            if (s[i + 1] == 'n') { out += '\n'; ++i; continue; }
            if (s[i + 1] == '\\') { out += '\\'; ++i; continue; }
        }
        out += s[i];
    }
    return out;
}

// =======================================================================
//  DOCUMENT + VECTOR STORE
// =======================================================================

struct Document {
    long long id;
    string text;
    vector<double> embedding;
};

struct ScoredDocument {
    const Document *doc;
    double score;
};

class VectorStore {
public:
    VectorStore(int dim = EMBEDDING_DIM) : dim_(dim), next_id_(1) {}

    // Add a single piece of text; embeds it and stores it in memory.
    long long addDocument(const string &text) {
        Document d;
        d.id = next_id_++;
        d.text = text;
        d.embedding = embed_query(text, dim_);
        documents_.push_back(std::move(d));
        return documents_.back().id;
    }

    // Bulk-import: one document per line from a plain text file.
    // Returns the number of documents added.
    size_t importFromTextFile(const string &path) {
        ifstream in(path);
        if (!in) {
            cerr << "Could not open file: " << path << "\n";
            return 0;
        }
        size_t count = 0;
        string line;
        while (getline(in, line)) {
            if (line.empty()) continue;
            addDocument(line);
            ++count;
        }
        return count;
    }

    size_t size() const { return documents_.size(); }

    // Brute-force nearest-neighbor search by cosine similarity.
    // Good up to roughly hundreds of thousands of documents; beyond that
    // you'd want an approximate-nearest-neighbor index (see note at bottom).
    vector<ScoredDocument> search(const string &query, size_t topK) const {
        vector<double> qvec = embed_query(query, dim_);
        vector<ScoredDocument> scored;
        scored.reserve(documents_.size());
        for (const auto &doc : documents_) {
            double sim = cosine_similarity(qvec, doc.embedding);
            scored.push_back({&doc, sim});
        }
        size_t k = min(topK, scored.size());
        partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                      [](const ScoredDocument &a, const ScoredDocument &b) {
                          return a.score > b.score;
                      });
        scored.resize(k);
        return scored;
    }

    // Persist all documents + their embeddings to a simple text file
    // so a "huge amount" of data survives between program runs without
    // needing to re-embed everything on load.
    bool saveToFile(const string &path) const {
        ofstream out(path, ios::trunc);
        if (!out) {
            cerr << "Could not open file for writing: " << path << "\n";
            return false;
        }
        out << documents_.size() << " " << dim_ << "\n";
        for (const auto &doc : documents_) {
            out << doc.id << "\n";
            out << escape_text(doc.text) << "\n";
            out << fixed << setprecision(10);
            for (size_t i = 0; i < doc.embedding.size(); ++i) {
                out << doc.embedding[i];
                if (i + 1 < doc.embedding.size()) out << " ";
            }
            out << "\n";
        }
        return true;
    }

    bool loadFromFile(const string &path) {
        ifstream in(path);
        if (!in) {
            cerr << "Could not open file for reading: " << path << "\n";
            return false;
        }
        size_t count;
        int dim;
        in >> count >> dim;
        in.ignore(); // consume trailing newline
        documents_.clear();
        documents_.reserve(count);
        dim_ = dim;
        long long max_id = 0;
        for (size_t i = 0; i < count; ++i) {
            Document d;
            string idLine, textLine, vecLine;
            if (!getline(in, idLine)) break;
            d.id = stoll(idLine);
            if (!getline(in, textLine)) break;
            d.text = unescape_text(textLine);
            if (!getline(in, vecLine)) break;
            istringstream iss(vecLine);
            d.embedding.resize(dim_);
            for (int j = 0; j < dim_; ++j) iss >> d.embedding[j];
            max_id = max(max_id, d.id);
            documents_.push_back(std::move(d));
        }
        next_id_ = max_id + 1;
        return true;
    }

private:
    int dim_;
    long long next_id_;
    vector<Document> documents_;
};

// =======================================================================
//  SMALL INTERACTIVE SHELL
// =======================================================================

void printHelp() {
    cout <<
        "Commands:\n"
        "  add <text>            add a single document\n"
        "  import <file>         add one document per line from a text file\n"
        "  search <text>         search top matches (default k=5)\n"
        "  search <text> | k=N   search top N matches\n"
        "  save <file>           save the store (text + embeddings) to disk\n"
        "  load <file>           load a previously saved store from disk\n"
        "  count                 show number of stored documents\n"
        "  help                  show this message\n"
        "  exit                  quit\n";
}

int main() {
    VectorStore store;
    cout << "Mini vector database (embedding dim = " << EMBEDDING_DIM << ")\n";
    printHelp();

    string line;
    while (true) {
        cout << "\n> ";
        if (!getline(cin, line)) break;
        if (line.empty()) continue;

        istringstream iss(line);
        string cmd;
        iss >> cmd;
        string rest;
        getline(iss, rest);
        if (!rest.empty() && rest[0] == ' ') rest = rest.substr(1);

        if (cmd == "exit" || cmd == "quit") {
            break;
        } else if (cmd == "help") {
            printHelp();
        } else if (cmd == "add") {
            if (rest.empty()) { cout << "Usage: add <text>\n"; continue; }
            long long id = store.addDocument(rest);
            cout << "Added document #" << id << " (" << store.size() << " total)\n";
        } else if (cmd == "import") {
            if (rest.empty()) { cout << "Usage: import <file>\n"; continue; }
            size_t n = store.importFromTextFile(rest);
            cout << "Imported " << n << " documents from \"" << rest
                 << "\" (" << store.size() << " total)\n";
        } else if (cmd == "search") {
            if (rest.empty()) { cout << "Usage: search <text> [| k=N]\n"; continue; }
            size_t k = 5;
            string query = rest;
            size_t pipePos = rest.find('|');
            if (pipePos != string::npos) {
                query = rest.substr(0, pipePos);
                string opt = rest.substr(pipePos + 1);
                size_t eqPos = opt.find("k=");
                if (eqPos != string::npos) {
                    try { k = stoul(opt.substr(eqPos + 2)); } catch (...) {}
                }
                // trim trailing space on query
                while (!query.empty() && query.back() == ' ') query.pop_back();
            }
            if (store.size() == 0) { cout << "Store is empty.\n"; continue; }
            auto results = store.search(query, k);
            cout << "Top " << results.size() << " result(s) for \"" << query << "\":\n";
            int rank = 1;
            for (const auto &r : results) {
                cout << "  " << rank++ << ". [score=" << fixed << setprecision(4)
                     << r.score << "] (id=" << r.doc->id << ") "
                     << r.doc->text << "\n";
            }
        } else if (cmd == "save") {
            if (rest.empty()) { cout << "Usage: save <file>\n"; continue; }
            if (store.saveToFile(rest)) cout << "Saved " << store.size() << " documents to \"" << rest << "\"\n";
        } else if (cmd == "load") {
            if (rest.empty()) { cout << "Usage: load <file>\n"; continue; }
            if (store.loadFromFile(rest)) cout << "Loaded " << store.size() << " documents from \"" << rest << "\"\n";
        } else if (cmd == "count") {
            cout << "Documents stored: " << store.size() << "\n";
        } else {
            cout << "Unknown command. Type 'help' for a list of commands.\n";
        }
    }

    cout << "Goodbye.\n";
    return 0;
}

// -----------------------------------------------------------------------
// Notes on scaling this to a "real" vector database:
//  - This version does a brute-force O(N) scan per search, which is fine
//    up to roughly hundreds of thousands of short documents on a laptop.
//  - For millions of documents / low-latency search, you'd replace the
//    linear scan in VectorStore::search() with an approximate nearest
//    neighbor index (e.g. HNSW, IVF, or a library like FAISS/Annoy/usearch).
//  - The embedding here is a toy hash-based embedding for demonstration.
//    For real semantic search you'd swap word_to_vector()/embed_query()
//    for a proper trained embedding model (e.g. via an API call).
// -----------------------------------------------------------------------