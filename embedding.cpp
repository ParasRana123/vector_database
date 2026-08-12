#include <iostream>
#include <string>
#include <vector>
#include <cctype>
#include <cmath>
#include <random>
#include <sstream>
#include <functional>
#include <iomanip>
using namespace std;

// Dimensionality of the output embedding vector.
// (Real models commonly use 384, 768, 1536, etc. We use a smaller
//  default here so the output is easy to read; change as needed.)
static const int EMBEDDING_DIM = 64;

// ---------------------------------------------------------------------
// Tokenizer: lowercases the text and splits it into alphanumeric words,
// stripping punctuation.
// ---------------------------------------------------------------------
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
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

// ---------------------------------------------------------------------
// Deterministically generate a pseudo-random embedding vector for a
// single word. The word's hash seeds a Mersenne Twister RNG, so the
// same word always yields the identical vector (acts like a fixed,
// pre-computed embedding lookup table without needing to store one).
// ---------------------------------------------------------------------
vector<double> word_to_vector(const string &word, int dim) {
    hash<string> hasher;
    size_t seed = hasher(word);
    mt19937_64 rng(seed);
    normal_distribution<double> dist(0.0, 1.0);
    vector<double> vec(dim);
    for (int i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

// ---------------------------------------------------------------------
// L2-normalize a vector in place (common preprocessing step so that
// cosine similarity reduces to a dot product).
// ---------------------------------------------------------------------
void l2_normalize(vector<double> &vec) {
    double norm = 0.0;
    for (double v : vec) norm += v * v;
    norm = sqrt(norm);
    if (norm > 1e-12) {
        for (double &v : vec) v /= norm;
    }
}

// ---------------------------------------------------------------------
// Convert a full text query into a single embedding vector by averaging
// the per-word vectors, then L2-normalizing the result.
// ---------------------------------------------------------------------
vector<double> embed_query(const string &query, int dim = EMBEDDING_DIM) {
    vector<string> tokens = tokenize(query);
    vector<double> embedding(dim, 0.0);
    if (tokens.empty()) {
        return embedding;
    }
    for (const auto &token : tokens) {
        std::vector<double> wv = word_to_vector(token, dim);
        for (int i = 0; i < dim; ++i) {
            embedding[i] += wv[i];
        }
    }
    for (double &v : embedding) {
        v /= static_cast<double>(tokens.size());
    }
    l2_normalize(embedding);
    return embedding;
}

void print_vector(const vector<double> &vec) {
    cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        cout << fixed << setprecision(5) << vec[i];
        if (i + 1 < vec.size()) cout << ", ";
    }
    cout << "]\n";
}

int main() {
    cout << "Enter a query: ";
    string query;
    getline(cin, query);
    vector<double> embedding = embed_query(query);
    cout << "\nQuery: \"" << query << "\"\n";
    cout << "Embedding vector (" << embedding.size() << " dims):\n";
    print_vector(embedding);
    return 0;
}