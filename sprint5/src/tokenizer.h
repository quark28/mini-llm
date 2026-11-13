#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
public:
    explicit Tokenizer(const std::filesystem::path& path);
    std::vector<int64_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int64_t>& ids, bool skip_special = true) const;

private:
    struct Special {
        std::string content;
        int64_t id;
    };

    void encode_plain(const std::string& text, std::vector<int64_t>& out) const;
    void encode_piece(const std::string& piece, std::vector<int64_t>& out) const;

    std::unordered_map<std::string, int64_t> vocab;
    std::unordered_map<std::string, int> merge_ranks;
    std::vector<std::string> id_to_token;
    std::vector<bool> is_special;
    std::vector<Special> specials;
    std::string byte_to_char[256];
    std::unordered_map<uint32_t, uint8_t> char_to_byte;
};