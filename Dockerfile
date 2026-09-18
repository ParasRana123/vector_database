# Stage 1: Build C++ VectorDB with CMake & GCC
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    tar \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Download & extract Linux ONNX Runtime binaries if needed
RUN if [ ! -f /app/third_party/onnxruntime/lib/libonnxruntime.so ]; then \
    mkdir -p /tmp/ort && \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz -O /tmp/ort/ort.tgz && \
    tar -xzf /tmp/ort/ort.tgz -C /tmp/ort && \
    mkdir -p /app/third_party/onnxruntime/include /app/third_party/onnxruntime/lib && \
    cp -r /tmp/ort/onnxruntime-linux-x64-1.17.1/include/* /app/third_party/onnxruntime/include/ && \
    cp /tmp/ort/onnxruntime-linux-x64-1.17.1/lib/* /app/third_party/onnxruntime/lib/ && \
    rm -rf /tmp/ort; \
    fi

# Build C++ binaries
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release --target vectordb_server -j$(nproc)

# Stage 2: Minimal Runtime Image
FROM ubuntu:22.04
WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/build/vectordb_server /app/vectordb_server
COPY --from=builder /app/models /app/models
COPY --from=builder /app/third_party/onnxruntime/lib/libonnxruntime.so* /usr/lib/

ENV PORT=8080
EXPOSE 8080
VOLUME ["/app/vectordb_data"]

CMD ["/bin/sh", "-c", "/app/vectordb_server ${PORT:-8080}"]
