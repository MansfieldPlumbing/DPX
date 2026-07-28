#pragma once
#include <string>
#include <functional>

typedef void (*UiOutputCallback)(const char* text);
extern UiOutputCallback g_ui_callback;

void sys_stream_token(int token_id);
