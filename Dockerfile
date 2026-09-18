# =========================================================================
# Stage 1: Build C++ VectorDB Server on Linux (Lightweight & Fast)
# =========================================================================
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    tar \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# 1. Download Linux x86_64 ONNX Runtime 1.17.1 library & headers
RUN mkdir -p /app/third_party/onnxruntime/include /app/third_party/onnxruntime/lib && \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz -O /tmp/ort.tgz && \
    tar -xzf /tmp/ort.tgz -C /tmp && \
    cp -r /tmp/onnxruntime-linux-x64-1.17.1/include/* /app/third_party/onnxruntime/include/ && \
    cp /tmp/onnxruntime-linux-x64-1.17.1/lib/* /app/third_party/onnxruntime/lib/ && \
    rm -rf /tmp/ort*

# 2. Download all-MiniLM-L6-v2 ONNX model and tokenizer if not already present
RUN if [ ! -f /app/models/all-MiniLM-L6-v2/tokenizer.json ]; then \
    mkdir -p /app/models/all-MiniLM-L6-v2/onnx && \
    wget -q https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/main/tokenizer.json -O /app/models/all-MiniLM-L6-v2/tokenizer.json && \
    wget -q https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx -O /app/models/all-MiniLM-L6-v2/onnx/all-MiniLM-L6-v2.onnx; \
    fi

# 3. Build C++ VectorDB server binary
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release --target vectordb_server -j$(nproc)

# =========================================================================
# Stage 2: Minimal Production Runtime
# =========================================================================
FROM ubuntu:22.04
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/build/vectordb_server /app/vectordb_server
COPY --from=builder /app/models /app/models
COPY --from=builder /app/third_party/onnxruntime/lib/libonnxruntime.so* /usr/lib/
COPY --from=builder /app/third_party/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/
RUN ldconfig

ENV PORT=8080
EXPOSE 8080
VOLUME ["/app/vectordb_data"]

CMD ["/bin/sh", "-c", "/app/vectordb_server ${PORT:-8080}"]
