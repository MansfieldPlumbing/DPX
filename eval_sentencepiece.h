#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <string_view>

struct SPToken {
    int id; float score; bool is_control;
};

class SentencePieceFastUnigram {
private:
    std::unordered_map<std::string_view, SPToken> vocab_table;
    std::vector<std::string> id_to_piece;
    int max_piece_bytes = 0;
public:
    void init_vocab(const std::vector<std::string>& pieces);
    void load_from_file(const char* path);
    std::vector<int> encode(const std::string& input_text);
    std::string detokenize(const std::vector<int>& ids);
};
