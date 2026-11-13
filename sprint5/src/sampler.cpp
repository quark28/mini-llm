#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace {

    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    void softmax(std::vector<float>& x) {
        float m = *std::max_element(x.begin(), x.end());
        double sum = 0.0;
        for (float& v : x) {
            v = std::exp(v - m);
            sum += v;
        }
        float total = static_cast<float>(sum);
        for (float& v : x) {
            v /= total;
        }
    }

    int64_t argmax(const std::vector<float>& x) {
        return std::max_element(x.begin(), x.end()) - x.begin();
    }

    bool contains(const std::vector<int64_t>& seq, int64_t token) {
        return std::find(seq.begin(), seq.end(), token) != seq.end();
    }

}

Sampler::Sampler(Model& model, SamplerSettings settings, unsigned seed)
    : model(model), settings(settings), rng(seed) {
}

Sampler::NGramTables Sampler::build_ngrams(const std::vector<int64_t>& seq, int n) {
    NGramTables tables;
    for (int i = 0; i + n <= static_cast<int>(seq.size()); ++i) {
        std::vector<int64_t> right(seq.begin() + i, seq.begin() + i + n - 1);
        std::vector<int64_t> left(seq.begin() + i + 1, seq.begin() + i + n);
        int64_t right_target = seq[i + n - 1];
        int64_t left_target = seq[i];

        if (!contains(right, PAD_ID) && right_target != PAD_ID) {
            tables.next[right].insert(right_target);
        }
        if (!contains(left, PAD_ID) && left_target != PAD_ID) {
            tables.prev[left].insert(left_target);
        }
    }
    return tables;
}

void Sampler::collect_penalties(const NGramTables& tables, int n, const std::vector<int64_t>& answer,
    int len, std::vector<std::vector<int64_t>>& penalties) {
    for (int i = 0; i + n <= len; ++i) {
        std::vector<int64_t> to_end(answer.begin() + i, answer.begin() + i + n - 1);
        std::vector<int64_t> to_start(answer.begin() + i + 1, answer.begin() + i + n);

        auto next = tables.next.find(to_end);
        if (next != tables.next.end() && !contains(to_end, MASK_ID)) {
            for (int64_t word : next->second) {
                penalties[i + n - 1].push_back(word);
            }
        }
        auto prev = tables.prev.find(to_start);
        if (prev != tables.prev.end() && !contains(to_start, MASK_ID)) {
            for (int64_t word : prev->second) {
                penalties[i].push_back(word);
            }
        }
    }
}

std::vector<int> Sampler::length_candidates() {
    std::vector<int64_t> masked(ANSWER_LEN, MASK_ID);
    Logits logits = model.decode(masked);

    double expected = 0.0;
    std::vector<float> probs(VOCAB);
    for (int p = 0; p < ANSWER_LEN; ++p) {
        const float* r = logits.row(0, p);
        probs.assign(r, r + VOCAB);
        softmax(probs);
        expected += 1.0f - probs[EOS_ID] - probs[PAD_ID];
    }
    float len_mean = static_cast<float>(expected) * settings.mean_length_scale;

    // same scale points as torch.linspace(0.8, 1.2, n)
    int n = settings.n_candidates;
    std::vector<float> scales;
    if (n == 1) {
        scales.push_back(1.0f);
    }
    else {
        float start = 0.8f;
        float end = 1.2f;
        float step = (end - start) / static_cast<float>(n - 1);
        for (int i = 0; i < n; ++i) {
            scales.push_back(i < n / 2 ? start + step * i : end - step * (n - i - 1));
        }
    }

    std::vector<int> lengths;
    for (float s : scales) {
        // nearbyint rounds halves to even, like torch.round
        int len = static_cast<int>(std::nearbyint(len_mean * s));
        lengths.push_back(std::clamp(len, settings.min_len, ANSWER_LEN - 1));
    }
    return lengths;
}

int64_t Sampler::pick(const std::vector<float>& probs) {
    if (settings.greedy) {
        return argmax(probs);
    }
    std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

void Sampler::fill_block(std::vector<int64_t>& answer, int start, int end, int len, const StepCallback& on_step) {
    std::vector<float> row(VOCAB);

    while (true) {
        std::vector<int> open;
        for (int p = start; p < end; ++p) {
            if (answer[p] == MASK_ID) {
                open.push_back(p);
            }
        }
        if (open.empty()) {
            break;
        }

        Logits logits = model.decode(answer);

        std::set<int64_t> used;
        for (int64_t t : answer) {
            if (t != MASK_ID) {
                used.insert(t);
            }
        }

        std::vector<std::vector<int64_t>> penalties(ANSWER_LEN);
        collect_penalties(build_ngrams(answer, settings.ngram_ans_len), settings.ngram_ans_len, answer, len, penalties);
        collect_penalties(user_ngrams, settings.ngram_user_len, answer, len, penalties);

        int best_pos = -1;
        int64_t best_token = 0;
        float best_conf = -1.0f;

        // only rows that can be opened matter, the rest lose to them anyway
        for (int p : open) {
            const float* r = logits.row(0, p);
            row.assign(r, r + VOCAB);

            row[MASK_ID] = row[PAD_ID] = row[EOS_ID] = NEG_INF;
            for (int64_t t : used) {
                row[t] = row[t] > 0 ? row[t] / settings.repeat_penalty : row[t] * settings.repeat_penalty;
            }
            for (int64_t t : penalties[p]) {
                row[t] -= settings.ngram_penalty;
            }
            for (float& v : row) {
                v /= settings.temperature;
            }

            std::vector<float> sorted = row;
            int k = std::min(settings.top_k, VOCAB);
            std::nth_element(sorted.begin(), sorted.begin() + (k - 1), sorted.end(), std::greater<float>());
            float kth = sorted[k - 1];
            for (float& v : row) {
                if (v < kth) {
                    v = NEG_INF;
                }
            }

            softmax(row);
            int64_t token = pick(row);
            if (row[token] > best_conf) {
                best_conf = row[token];
                best_pos = p;
                best_token = token;
            }
        }

        answer[best_pos] = best_token;
        if (on_step) {
            on_step(answer);
        }
    }
}

std::vector<Sampler::LooRow> Sampler::leave_one_out(const std::vector<int64_t>& answer, int start, int end) {
    int k = end - start;
    std::vector<int64_t> batch;
    batch.reserve(static_cast<size_t>(k) * ANSWER_LEN);
    for (int r = 0; r < k; ++r) {
        batch.insert(batch.end(), answer.begin(), answer.end());
        batch[static_cast<size_t>(r) * ANSWER_LEN + start + r] = MASK_ID;
    }

    Logits logits = model.decode(batch);

    std::vector<LooRow> rows;
    std::vector<float> probs(VOCAB);
    for (int r = 0; r < k; ++r) {
        const float* x = logits.row(r, start + r);
        probs.assign(x, x + VOCAB);
        probs[MASK_ID] = probs[PAD_ID] = probs[EOS_ID] = NEG_INF;
        softmax(probs);

        int64_t best = argmax(probs);
        rows.push_back({ probs[answer[start + r]], probs[best], best });
    }
    return rows;
}

float Sampler::gstar(std::vector<int64_t>& answer, int start, int end, int rounds, bool score,
    const StepCallback& on_step) {
    for (int step = 0; step < rounds; ++step) {
        std::vector<LooRow> rows = leave_one_out(answer, start, end);

        int replace_pos = -1;
        int64_t replace_token = 0;
        double best_gap = -std::numeric_limits<double>::infinity();

        for (int r = 0; r < static_cast<int>(rows.size()); ++r) {
            int64_t cur = answer[start + r];
            bool is_bad = rows[r].p_cur < settings.margin * rows[r].p_best;
            bool is_other = rows[r].best_token != cur;
            bool is_free = last_user_phrase.count(cur) == 0;
            double gap = rows[r].p_best - rows[r].p_cur;
            if (is_bad && is_other && is_free && gap > best_gap) {
                replace_pos = start + r;
                replace_token = rows[r].best_token;
                best_gap = gap;
            }
        }

        if (replace_pos < 0) {
            break;
        }
        answer[replace_pos] = replace_token;
        if (on_step) {
            on_step(answer);
        }
    }

    if (!score) {
        return 0.0f;
    }
    std::vector<LooRow> rows = leave_one_out(answer, start, end);
    double sum = 0.0;
    for (const LooRow& row : rows) {
        sum += std::log(std::max(row.p_cur, 1e-9f));
    }
    return static_cast<float>(sum / rows.size());
}

std::vector<int64_t> Sampler::generate(std::vector<int64_t>& context_ids, const StepCallback& on_step) {
    model.encode(context_ids);

    user_ngrams = build_ngrams(context_ids, settings.ngram_user_len);

    // tokens of the last user message are never replaced by gstar
    int last_user = -1;
    int last_bot = 0;
    for (int i = 0; i < static_cast<int>(context_ids.size()); ++i) {
        if (context_ids[i] == USER_ID) {
            last_user = i;
        }
        if (context_ids[i] == BOT_ID) {
            last_bot = i;
        }
    }
    last_user_phrase = std::set<int64_t>(context_ids.begin() + last_user + 1, context_ids.begin() + last_bot);

    std::vector<int> lengths = length_candidates();

    std::vector<int64_t> best;
    float best_score = NEG_INF;

    for (int len : lengths) {
        std::vector<int64_t> answer(ANSWER_LEN, MASK_ID);
        answer[len] = EOS_ID;
        std::fill(answer.begin() + len + 1, answer.end(), PAD_ID);

        for (int start = 0; start < len; start += settings.block_len) {
            int end = std::min(start + settings.block_len, len);
            fill_block(answer, start, end, len, on_step);
            gstar(answer, start, end, settings.block_rounds, false, on_step);
        }

        float score = gstar(answer, 0, len, settings.global_rounds, true, on_step);
        if (score > best_score) {
            best_score = score;
            best = answer;
        }
    }
    return best;
}