#pragma once
#include "model.h"

#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <vector>

constexpr int64_t PAD_ID = 0;
constexpr int64_t MASK_ID = 1;
constexpr int64_t USER_ID = 2;
constexpr int64_t BOT_ID = 3;
constexpr int64_t EOS_ID = 4;

struct SamplerSettings {
    float mean_length_scale = 1.0f;
    int min_len = 3;
    int n_candidates = 4;
    int block_len = 8;
    float repeat_penalty = 1.05f;
    int ngram_ans_len = 3;
    int ngram_user_len = 4;
    float ngram_penalty = 5.0f;
    float temperature = 0.5f;
    int top_k = 50;
    float margin = 0.5f;
    int block_rounds = 2;
    int global_rounds = 2;
    bool greedy = false;
};

using StepCallback = std::function<void(const std::vector<int64_t>& answer)>;

class Sampler {
public:
    Sampler(Model& model, SamplerSettings settings, unsigned seed);
    std::vector<int64_t> generate(std::vector<int64_t>& context_ids, const StepCallback& on_step = nullptr);

private:
    using NGrams = std::map<std::vector<int64_t>, std::set<int64_t>>;

    struct NGramTables {
        NGrams next;
        NGrams prev;
    };

    struct LooRow {
        float p_cur;
        float p_best;
        int64_t best_token;
    };

    static NGramTables build_ngrams(const std::vector<int64_t>& seq, int n);
    static void collect_penalties(const NGramTables& tables, int n, const std::vector<int64_t>& answer,
        int len, std::vector<std::vector<int64_t>>& penalties);

    std::vector<int> length_candidates();
    void fill_block(std::vector<int64_t>& answer, int start, int end, int len, const StepCallback& on_step);
    std::vector<LooRow> leave_one_out(const std::vector<int64_t>& answer, int start, int end);
    float gstar(std::vector<int64_t>& answer, int start, int end, int rounds, bool score, const StepCallback& on_step);
    int64_t pick(const std::vector<float>& probs);

    Model& model;
    SamplerSettings settings;
    std::mt19937 rng;
    NGramTables user_ngrams;
    std::set<int64_t> last_user_phrase;
};