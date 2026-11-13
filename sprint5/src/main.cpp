#include "console.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

    constexpr int THREADS = 4;

    double seconds_since(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    std::string format_seconds(double s) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.1f s", s);
        return buffer;
    }

    std::vector<int64_t> until_eos(const std::vector<int64_t>& answer) {
        auto eos = std::find(answer.begin(), answer.end(), EOS_ID);
        return std::vector<int64_t>(answer.begin(), eos);
    }

    // same cleanup the SFT data got in Sprint 1
    std::string normalize(const std::string& text) {
        static const std::pair<std::string, std::string> replacements[] = {
            {"\xE2\x80\x98", "'"}, {"\xE2\x80\x99", "'"}, {"\xE2\x80\x9A", "'"}, {"\xE2\x80\x9B", "'"},
            {"\xCA\xBC", "'"},
            {"\xE2\x80\x9C", "\""}, {"\xE2\x80\x9D", "\""}, {"\xE2\x80\x9E", "\""}, {"\xE2\x80\x9F", "\""},
            {"\xC2\xA0", " "}, {"\r", ""},
        };
        std::string s = text;
        for (const auto& [from, to] : replacements) {
            for (size_t pos = s.find(from); pos != std::string::npos; pos = s.find(from, pos + to.size())) {
                s.replace(pos, from.size(), to);
            }
        }

        std::string out;
        for (char c : s) {
            if (c == ' ' && !out.empty() && out.back() == ' ') {
                continue;
            }
            out += c;
        }
        size_t first = out.find_first_not_of(" \t\n");
        if (first == std::string::npos) {
            return "";
        }
        size_t last = out.find_last_not_of(" \t\n");
        return out.substr(first, last - first + 1);
    }

    // cut old turns from the left so the window starts on a user turn
    std::vector<int64_t> trim_history(const std::vector<int64_t>& ids) {
        if (static_cast<int>(ids.size()) <= CONTEXT_LEN) {
            return ids;
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == USER_ID && ids.size() - i <= CONTEXT_LEN) {
                return std::vector<int64_t>(ids.begin() + i, ids.end());
            }
        }
        return std::vector<int64_t>(ids.end() - CONTEXT_LEN, ids.end());
    }

    std::vector<int64_t> build_context(const std::vector<int64_t>& history) {
        std::vector<int64_t> window = trim_history(history);
        std::vector<int64_t> context(CONTEXT_LEN - window.size(), PAD_ID);
        context.insert(context.end(), window.begin(), window.end());
        return context;
    }

    // revealed tokens as text, still masked ones as blocks
    std::string render(const Tokenizer& tok, const std::vector<int64_t>& answer) {
        std::string text;
        std::vector<int64_t> group;
        for (int64_t t : answer) {
            if (t == EOS_ID) {
                break;
            }
            if (t == MASK_ID) {
                text += tok.decode(group);
                group.clear();
                text += "\xE2\x96\x91";
            }
            else {
                group.push_back(t);
            }
        }
        return text + tok.decode(group);
    }

    std::string last_chars(const std::string& text, size_t max_chars) {
        std::vector<size_t> starts;
        for (size_t i = 0; i < text.size(); ++i) {
            if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) {
                starts.push_back(i);
            }
        }
        if (starts.size() <= max_chars) {
            return text;
        }
        return "\xE2\x80\xA6" + text.substr(starts[starts.size() - max_chars + 1]);
    }

    int run_golden(Model& model, const Tokenizer& tok) {
        std::ifstream file(std::filesystem::path(MODELS_DIR) / "golden_gen.json");
        if (!file) {
            std::cerr << "golden_gen.json not found in " << MODELS_DIR << "\n";
            return 1;
        }
        nlohmann::json golden = nlohmann::json::parse(file);

        SamplerSettings settings;
        settings.greedy = true;
        Sampler sampler(model, settings, 0);

        int equal = 0;
        double total = 0.0;
        for (size_t k = 0; k < golden.size(); ++k) {
            std::vector<int64_t> context = golden[k]["history_ids"].get<std::vector<int64_t>>();
            std::vector<int64_t> expected = golden[k]["reply_ids"].get<std::vector<int64_t>>();

            auto start = std::chrono::steady_clock::now();
            std::vector<int64_t> answer = sampler.generate(context);
            double took = seconds_since(start);
            total += took;

            std::string text = tok.decode(until_eos(answer));
            bool same = answer == expected;
            if (same) {
                ++equal;
            }

            console::write("[" + std::to_string(k + 1) + "/" + std::to_string(golden.size()) + "] " +
                (same ? "ok   " : "DIFF ") + format_seconds(took) + "  " + text + "\n");
            if (!same) {
                size_t pos = 0;
                while (pos < answer.size() && answer[pos] == expected[pos]) ++pos;
                console::write("      expected: " + golden[k]["reply_text"].get<std::string>() +
                    "\n      first difference at position " + std::to_string(pos) + "\n");
            }
        }

        console::write("Equal " + std::to_string(equal) + "/" + std::to_string(golden.size()) +
            ", average " + format_seconds(total / golden.size()) + " per reply\n");
        return equal == static_cast<int>(golden.size()) ? 0 : 1;
    }

    int run_chat(Model& model, const Tokenizer& tok) {
        SamplerSettings settings;
        Sampler sampler(model, settings, std::random_device{}());

        console::write("mini-llm classic SFT. Type 'exit' to quit, 'reset' to clear the history.\n\n");

        std::vector<int64_t> history;
        std::string line;
        while (true) {
            console::write("USER: ");
            if (!console::read_line(line)) {
                break;
            }
            std::string message = normalize(line);
            if (message == "exit") {
                break;
            }
            if (message == "reset") {
                history.clear();
                console::write("(history cleared)\n\n");
                continue;
            }
            if (message.empty()) {
                continue;
            }

            std::vector<int64_t> user_ids = tok.encode(message);
            history.push_back(USER_ID);
            history.insert(history.end(), user_ids.begin(), user_ids.end());
            history.push_back(BOT_ID);
            std::vector<int64_t> context = build_context(history);

            int candidate = 0;
            int last_revealed = 0;
            size_t room = static_cast<size_t>(std::max(console::width() - 16, 10));

            // a new candidate starts from an all masked answer, so the revealed count drops
            auto show = [&](const std::vector<int64_t>& answer) {
                int revealed = 0;
                for (int64_t t : answer) {
                    revealed += (t != MASK_ID && t != EOS_ID && t != PAD_ID);
                }
                if (revealed < last_revealed || candidate == 0) {
                    ++candidate;
                }
                last_revealed = revealed;
                std::string tag = " BOT [" + std::to_string(candidate) + "/" + std::to_string(settings.n_candidates) + "]: ";
                console::write("\r\x1b[2K" + tag + last_chars(render(tok, answer), room));
                };

            auto start = std::chrono::steady_clock::now();
            std::vector<int64_t> answer = sampler.generate(context, show);
            double took = seconds_since(start);

            std::vector<int64_t> reply = until_eos(answer);
            console::write("\r\x1b[2K BOT: " + tok.decode(reply) + "  \x1b[2m(" + format_seconds(took) + ")\x1b[0m\n\n");

            history.insert(history.end(), reply.begin(), reply.end());
            history.push_back(EOS_ID);
        }
        return 0;
    }

}

int main(int argc, char** argv) {
    console::init();
    try {
        std::filesystem::path dir = MODELS_DIR;
        console::write("Loading model...\n");
        Model model(dir, THREADS);
        Tokenizer tok(dir / "tokenizer.json");

        if (argc > 1 && std::string(argv[1]) == "--golden") {
            return run_golden(model, tok);
        }
        return run_chat(model, tok);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ORT error: " << e.what() << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 1;
}