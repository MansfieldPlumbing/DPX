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
    }
}
