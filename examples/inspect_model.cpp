#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {

    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "MiniLM"
    );

    Ort::SessionOptions session_options;

    session_options.SetIntraOpNumThreads(1);

    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    // Windows requires a wide-character model path
    const wchar_t* model_path =
        L"../models/all-MiniLM-L6-v2/onnx/all-MiniLM-L6-v2.onnx";

    try {

        Ort::Session session(
            env,
            model_path,
            session_options
        );

        Ort::AllocatorWithDefaultOptions allocator;

        size_t input_count =
            session.GetInputCount();

        size_t output_count =
            session.GetOutputCount();

        std::cout << "Model loaded successfully!\n\n";

        std::cout << "Number of inputs: "
                  << input_count << "\n";

        std::cout << "Number of outputs: "
                  << output_count << "\n\n";


        // Print inputs
        std::cout << "===== INPUTS =====\n";

        for (size_t i = 0; i < input_count; i++) {

            auto name =
                session.GetInputNameAllocated(
                    i,
                    allocator
                );

            std::cout << "Input " << i
                      << ": "
                      << name.get()
                      << "\n";
        }


        // Print outputs
        std::cout << "\n===== OUTPUTS =====\n";

        for (size_t i = 0; i < output_count; i++) {

            auto name =
                session.GetOutputNameAllocated(
                    i,
                    allocator
                );

            std::cout << "Output " << i
                      << ": "
                      << name.get()
                      << "\n";
        }

    }
    catch (const Ort::Exception& e) {

        std::cerr << "ONNX Runtime Error:\n";
        std::cerr << e.what() << "\n";

        return 1;
    }

    return 0;
}