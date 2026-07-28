#include "eval_sentencepiece.h"
#include <algorithm>
#include <fstream>
#include <iostream>

static uint64_t read_varint(const uint8_t*& p, const uint8_t* end) {
    uint64_t r = 0; int shift = 0;
    while (p < end) {
        uint8_t b = *p++;
        r |= static_cast<uint64_t>(b & 0x7F) << shift;
        if (!(b & 0x80)) return r;
        shift += 7;
    }
    return r;
}

void SentencePieceFastUnigram::load_from_file(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::cerr << "\033[1;31m[ERROR] Failed to open SPM vocabulary: " << path << "\033[0m\n";
        return;
    }
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(size);
    f.read((char*)data.data(), size);
    
    const uint8_t* p = data.data();
    const uint8_t* end = p + size;
    std::vector<std::string> pieces;
    
    while (p < end) {
        uint64_t tag = read_varint(p, end);
        int field = tag >> 3;
        int wire = tag & 7;
        if (field == 1 && wire == 2) {
            uint64_t len = read_varint(p, end);
            const uint8_t* p_end = p + len;
            std::string piece_str;
            while (p < p_end) {
                uint64_t ptag = read_varint(p, p_end);
                int pfield = ptag >> 3;
                int pwire = ptag & 7;
                if (pfield == 1 && pwire == 2) {
                    uint64_t slen = read_varint(p, p_end);
                    piece_str = std::string((const char*)p, slen);
                    p += slen;
                } else if (pwire == 0) { read_varint(p, p_end); }
                else if (pwire == 1) { p += 8; }
                else if (pwire == 5) { p += 4; }
                else if (pwire == 2) { uint64_t slen = read_varint(p, p_end); p += slen; }
            }
            pieces.push_back(piece_str);
        } else if (wire == 0) { read_varint(p, end); }
        else if (wire == 1) { p += 8; }
        else if (wire == 5) { p += 4; }
        else if (wire == 2) { uint64_t slen = read_varint(p, end); p += slen; }
    }
    init_vocab(pieces);
}

void SentencePieceFastUnigram::init_vocab(const std::vector<std::string>& pieces) { 
    id_to_piece = pieces;
    vocab_table.clear(); // Safe re-entry
    // Map string_views directly to the persistent members of id_to_piece to prevent dangling pointer crashes
    for (size_t i = 0; i < id_to_piece.size(); i++) {
        vocab_table[id_to_piece[i]] = { (int)i, 0.0f, false };
        if ((int)id_to_piece[i].length() > max_piece_bytes) max_piece_bytes = (int)id_to_piece[i].length();
    }
}

std::vector<int> SentencePieceFastUnigram::encode(const std::string& input_text) {
    std::string norm = " " + input_text;
    std::vector<int> tokens;
    int len = norm.size();
    int pos = 0;

    while (pos < len) {
        int best_len = 0; int best_id = -1;
        int limit = (std::min)(len - pos, max_piece_bytes ? max_piece_bytes : 32);

        for (int l = limit; l >= 1; --l) {
            std::string_view slice(norm.data() + pos, l);
            auto it = vocab_table.find(slice);
            if (it != vocab_table.end() && !it->second.is_control) {
                best_len = l; best_id = it->second.id; break;
            }
        }
        if (best_id >= 0) {
            tokens.push_back(best_id);
            pos += best_len;
        } else {
            tokens.push_back(static_cast<uint8_t>(norm[pos]));
            pos += 1;
        }
    }
    return tokens;
}

std::string SentencePieceFastUnigram::detokenize(const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) {
        if (id < 0 || id >= (int)id_to_piece.size()) continue;
        std::string piece = id_to_piece[id];
        
        if (piece == "<bos>" || piece == "<eos>" || piece == "<pad>" || piece == "<unk>") continue;

        if (piece.length() == 6 && piece.substr(0, 3) == "<0x" && piece.back() == '>') {
            int hex_val = std::stoi(piece.substr(3, 2), nullptr, 16);
            result += static_cast<char>(hex_val);
        } else {
            result += piece;
        }
    }
    
    std::string target = "\xE2\x96\x81";
    size_t start_pos = 0;
    while((start_pos = result.find(target, start_pos)) != std::string::npos) {
        result.replace(start_pos, target.length(), " ");
        start_pos += 1;
    }

    if (!result.empty() && result[0] == ' ') {
        result = result.substr(1);
    }
    return result;
}
