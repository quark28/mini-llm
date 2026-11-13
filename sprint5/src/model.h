#pragma once
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <vector>

constexpr int CONTEXT_LEN = 256;
constexpr int ANSWER_LEN = 64;
constexpr int VOCAB = 12288;

struct Logits {
    Ort::Value value;
    int batch;
    const float* row(int b, int p) const;
};

class Model {
public:
    Model(const std::filesystem::path& dir, int threads);
    void encode(std::vector<int64_t>& context_ids);
    Logits decode(std::vector<int64_t>& answer_ids);

private:
    Ort::Env env;
    Ort::MemoryInfo cpu;
    Ort::Session encoder;
    Ort::Session decoder;
    std::vector<Ort::Value> inputs;
};