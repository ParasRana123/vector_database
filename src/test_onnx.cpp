#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    try {
        Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "VectorDB"
        );

        std::cout << "ONNX Runtime initialized successfully!\n";
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: "
                  << e.what() << "\n";

        return 1;
    }

    return 0;
}