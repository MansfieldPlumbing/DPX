#include "sys_token_streamer.h"
#include "eval_sentencepiece.h"
#include <iostream>

extern SentencePieceFastUnigram g_tokenizer;
extern UiOutputCallback g_ui_callback;

void sys_stream_token(int token_id) {
    std::string text = g_tokenizer.detokenize({token_id});
    if (!text.empty()) {
        std::cout << text << std::flush;
        if (g_ui_callback) g_ui_callback(text.c_str());
    } else {
        std::string piece = g_tokenizer.get_piece(token_id);
        if (!piece.empty() && piece != "<bos>" && piece != "<eos>" && piece != "<pad>" && piece != "<unk>") {
            std::cout << piece << std::flush;
            if (g_ui_callback) g_ui_callback(piece.c_str());
        }
    }
}
