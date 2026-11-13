#include "model.h"

#include <array>

namespace {

    const char* ENCODER_INPUTS[] = { "context_ids" };
    const char* ENCODER_OUTPUTS[] = { "memory", "mask" };
    const char* DECODER_INPUTS[] = { "memory", "mask", "answer_ids" };
    const char* DECODER_OUTPUTS[] = { "logits" };

    Ort::SessionOptions make_options(int threads) {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        return options;
    }

}

const float* Logits::row(int b, int p) const {
    size_t offset = (static_cast<size_t>(b) * ANSWER_LEN + p) * VOCAB;
    return value.GetTensorData<float>() + offset;
}

Model::Model(const std::filesystem::path& dir, int threads)
    : env(ORT_LOGGING_LEVEL_WARNING, "mini-llm"),
    cpu(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
    encoder(env, (dir / "encoder.onnx").c_str(), make_options(threads)),
    decoder(env, (dir / "decoder.onnx").c_str(), make_options(threads)) {
}

void Model::encode(std::vector<int64_t>& context_ids) {
    std::array<int64_t, 2> shape{ 1, CONTEXT_LEN };
    // wraps the caller's buffer without copying
    Ort::Value ids = Ort::Value::CreateTensor<int64_t>(
        cpu, context_ids.data(), context_ids.size(), shape.data(), shape.size());

    std::vector<Ort::Value> out = encoder.Run(
        Ort::RunOptions{ nullptr }, ENCODER_INPUTS, &ids, 1, ENCODER_OUTPUTS, 2);

    // decoder input order: memory, mask, answer_ids
    inputs.clear();
    inputs.push_back(std::move(out[0]));
    inputs.push_back(std::move(out[1]));
    inputs.emplace_back(nullptr);
}

Logits Model::decode(std::vector<int64_t>& answer_ids) {
    int batch = static_cast<int>(answer_ids.size()) / ANSWER_LEN;
    std::array<int64_t, 2> shape{ batch, ANSWER_LEN };
    inputs[2] = Ort::Value::CreateTensor<int64_t>(
        cpu, answer_ids.data(), answer_ids.size(), shape.data(), shape.size());

    // memory and mask stay at batch 1, the graph expands them itself
    std::vector<Ort::Value> out = decoder.Run(
        Ort::RunOptions{ nullptr }, DECODER_INPUTS, inputs.data(), 3, DECODER_OUTPUTS, 1);

    return Logits{ std::move(out[0]), batch };
}