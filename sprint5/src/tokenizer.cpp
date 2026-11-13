#include "tokenizer.h"
#include "unicode_tables.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace {

    enum class CharClass { Letter, Number, Space, Other };

    struct CodePoint {
        uint32_t value;
        size_t offset;
        size_t length;
    };

    bool in_ranges(uint32_t cp, const CodeRange* begin, const CodeRange* end) {
        const CodeRange* it = std::upper_bound(begin, end, cp,
            [](uint32_t value, const CodeRange& range) { return value < range.lo; });
        return it != begin && cp <= (it - 1)->hi;
    }

    bool is_space(uint32_t cp) {
        return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
            (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
            cp == 0x205F || cp == 0x3000;
    }

    CharClass classify(uint32_t cp) {
        if (is_space(cp)) {
            return CharClass::Space;
        }
        if (in_ranges(cp, std::begin(LETTER_RANGES), std::end(LETTER_RANGES))) {
            return CharClass::Letter;
        }
        if (in_ranges(cp, std::begin(NUMBER_RANGES), std::end(NUMBER_RANGES))) {
            return CharClass::Number;
        }
        return CharClass::Other;
    }

    std::string to_utf8(uint32_t cp) {
        std::string s;
        if (cp < 0x80) {
            s += static_cast<char>(cp);
        }
        else if (cp < 0x800) {
            s += static_cast<char>(0xC0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
        else if (cp < 0x10000) {
            s += static_cast<char>(0xE0 | (cp >> 12));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
        else {
            s += static_cast<char>(0xF0 | (cp >> 18));
            s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return s;
    }

    // length of a valid UTF-8 sequence at i, or how many bytes to drop if it is broken
    size_t utf8_step(const std::string& s, size_t i, bool& valid) {
        auto byte = [&](size_t k) { return static_cast<uint8_t>(s[k]); };
        uint8_t b = byte(i);
        valid = false;
        if (b < 0x80) {
            valid = true;
            return 1;
        }

        size_t need = 0;
        uint8_t lo = 0x80;
        uint8_t hi = 0xBF;
        if (b >= 0xC2 && b <= 0xDF) {
            need = 1;
        }
        else if (b >= 0xE0 && b <= 0xEF) {
            need = 2;
            if (b == 0xE0) lo = 0xA0;
            if (b == 0xED) hi = 0x9F;
        }
        else if (b >= 0xF0 && b <= 0xF4) {
            need = 3;
            if (b == 0xF0) lo = 0x90;
            if (b == 0xF4) hi = 0x8F;
        }
        else {
            return 1;
        }

        size_t k = 1;
        for (; k <= need; ++k) {
            if (i + k >= s.size()) {
                return k;
            }
            uint8_t c = byte(i + k);
            uint8_t first_lo = (k == 1) ? lo : 0x80;
            uint8_t first_hi = (k == 1) ? hi : 0xBF;
            if (c < first_lo || c > first_hi) {
                return k;
            }
        }
        valid = true;
        return need + 1;
    }

    uint32_t decode_cp(const std::string& s, size_t i, size_t len) {
        auto byte = [&](size_t k) { return static_cast<uint32_t>(static_cast<uint8_t>(s[k])); };
        if (len == 1) return byte(i);
        if (len == 2) return ((byte(i) & 0x1F) << 6) | (byte(i + 1) & 0x3F);
        if (len == 3) return ((byte(i) & 0x0F) << 12) | ((byte(i + 1) & 0x3F) << 6) | (byte(i + 2) & 0x3F);
        return ((byte(i) & 0x07) << 18) | ((byte(i + 1) & 0x3F) << 12) | ((byte(i + 2) & 0x3F) << 6) | (byte(i + 3) & 0x3F);
    }

    std::vector<CodePoint> to_code_points(const std::string& text) {
        std::vector<CodePoint> cps;
        size_t i = 0;
        while (i < text.size()) {
            bool valid = false;
            size_t len = utf8_step(text, i, valid);
            // a broken byte becomes U+FFFD, which is classified as Other
            cps.push_back({ valid ? decode_cp(text, i, len) : 0xFFFD, i, len });
            i += len;
        }
        return cps;
    }

    // what String::from_utf8_lossy does in the original tokenizer
    std::string utf8_lossy(const std::string& bytes) {
        std::string out;
        size_t i = 0;
        while (i < bytes.size()) {
            bool valid = false;
            size_t len = utf8_step(bytes, i, valid);
            if (valid) {
                out.append(bytes, i, len);
            }
            else {
                out += "\xEF\xBF\xBD";
            }
            i += len;
        }
        return out;
    }

    // GPT-2 pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    std::vector<std::string> pre_tokenize(const std::string& text) {
        std::vector<CodePoint> cps = to_code_points(text);
        std::vector<CharClass> cls;
        for (const CodePoint& cp : cps) {
            cls.push_back(classify(cp.value));
        }

        size_t n = cps.size();
        auto run_end = [&](size_t j, CharClass c) {
            while (j < n && cls[j] == c) ++j;
            return j;
            };

        std::vector<std::string> pieces;
        size_t i = 0;
        while (i < n) {
            size_t end = i;

            if (cps[i].value == '\'') {
                static const char* suffixes[] = { "s", "t", "re", "ve", "m", "ll", "d" };
                for (const char* suffix : suffixes) {
                    size_t len = std::char_traits<char>::length(suffix);
                    bool ok = i + len < n;
                    for (size_t k = 0; ok && k < len; ++k) {
                        ok = cps[i + 1 + k].value == static_cast<uint32_t>(suffix[k]);
                    }
                    if (ok) {
                        end = i + 1 + len;
                        break;
                    }
                }
            }

            if (end == i) {
                size_t start = (cps[i].value == ' ' && i + 1 < n && cls[i + 1] != CharClass::Space) ? i + 1 : i;
                if (cls[start] != CharClass::Space) {
                    end = run_end(start, cls[start]);
                }
            }

            if (end == i) {
                size_t spaces = run_end(i, CharClass::Space);
                // a space run gives its last space to the following word
                end = (spaces == n || spaces - i == 1) ? spaces : spaces - 1;
            }

            size_t from = cps[i].offset;
            size_t to = (end < n) ? cps[end].offset : text.size();
            pieces.push_back(text.substr(from, to - from));
            i = end;
        }
        return pieces;
    }

}

Tokenizer::Tokenizer(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path.string());
    }
    nlohmann::json json = nlohmann::json::parse(file);

    // GPT-2 byte to visible character table
    std::vector<int> visible;
    for (int b = '!'; b <= '~'; ++b) visible.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) visible.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) visible.push_back(b);
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        bool keeps_itself = std::find(visible.begin(), visible.end(), b) != visible.end();
        uint32_t cp = keeps_itself ? static_cast<uint32_t>(b) : 256u + extra++;
        byte_to_char[b] = to_utf8(cp);
        char_to_byte[cp] = static_cast<uint8_t>(b);
    }

    for (auto& [token, id] : json["model"]["vocab"].items()) {
        vocab[token] = id.get<int64_t>();
    }

    const nlohmann::json& merges = json["model"]["merges"];
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const nlohmann::json& m = merges[rank];
        // newer files store a pair, older ones store "left right"
        std::string key = m.is_string() ? m.get<std::string>()
            : m[0].get<std::string>() + " " + m[1].get<std::string>();
        merge_ranks.emplace(key, static_cast<int>(rank));
    }

    for (const nlohmann::json& t : json["added_tokens"]) {
        int64_t id = t["id"].get<int64_t>();
        std::string content = t["content"].get<std::string>();
        specials.push_back({ content, id });
        vocab[content] = id;
    }

    int64_t max_id = 0;
    for (const auto& [token, id] : vocab) {
        max_id = std::max(max_id, id);
    }
    id_to_token.resize(max_id + 1);
    is_special.assign(max_id + 1, false);
    for (const auto& [token, id] : vocab) {
        id_to_token[id] = token;
    }
    for (const nlohmann::json& t : json["added_tokens"]) {
        is_special[t["id"].get<int64_t>()] = t.value("special", false);
    }
}

std::vector<int64_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int64_t> out;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best_at = std::string::npos;
        const Special* best = nullptr;
        for (const Special& s : specials) {
            size_t at = text.find(s.content, pos);
            if (at == std::string::npos) continue;
            if (at < best_at || (at == best_at && s.content.size() > best->content.size())) {
                best_at = at;
                best = &s;
            }
        }
        if (!best) {
            encode_plain(text.substr(pos), out);
            break;
        }
        encode_plain(text.substr(pos, best_at - pos), out);
        out.push_back(best->id);
        pos = best_at + best->content.size();
    }
    return out;
}

void Tokenizer::encode_plain(const std::string& text, std::vector<int64_t>& out) const {
    for (const std::string& piece : pre_tokenize(text)) {
        encode_piece(piece, out);
    }
}

void Tokenizer::encode_piece(const std::string& piece, std::vector<int64_t>& out) const {
    std::vector<std::string> word;
    for (unsigned char b : piece) {
        word.push_back(byte_to_char[b]);
    }

    while (word.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_pos = 0;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = merge_ranks.find(word[i] + " " + word[i + 1]);
            if (it != merge_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }
        if (best_rank == std::numeric_limits<int>::max()) {
            break;
        }
        word[best_pos] += word[best_pos + 1];
        word.erase(word.begin() + best_pos + 1);
    }

    for (const std::string& symbol : word) {
        auto it = vocab.find(symbol);
        if (it != vocab.end()) {
            out.push_back(it->second);
        }
    }
}

std::string Tokenizer::decode(const std::vector<int64_t>& ids, bool skip_special) const {
    std::string bytes;
    for (int64_t id : ids) {
        if (id < 0 || id >= static_cast<int64_t>(id_to_token.size())) continue;
        if (skip_special && is_special[id]) continue;

        const std::string& token = id_to_token[id];
        std::string token_bytes;
        bool mapped = true;
        for (const CodePoint& cp : to_code_points(token)) {
            auto it = char_to_byte.find(cp.value);
            if (it == char_to_byte.end()) {
                mapped = false;
                break;
            }
            token_bytes += static_cast<char>(it->second);
        }
        bytes += mapped ? token_bytes : token;
    }
    return utf8_lossy(bytes);
}