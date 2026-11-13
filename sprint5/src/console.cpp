#include "console.h"

#include <cstdio>
#include <iostream>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace console {

#ifdef _WIN32

    namespace {

        std::wstring widen(const std::string& s) {
            if (s.empty()) return {};
            int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
            std::wstring w(n, L'\0');
            MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
            return w;
        }

        std::string narrow(const std::wstring& w) {
            if (w.empty()) return {};
            int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
            std::string s(n, '\0');
            WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
            return s;
        }

        bool is_console(HANDLE h) {
            DWORD mode = 0;
            return GetConsoleMode(h, &mode) != 0;
        }

    }

    void init() {
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
        HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (GetConsoleMode(out, &mode)) {
            // lets \r and escape codes redraw the line
            SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }

    void write(const std::string& text) {
        HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
        if (is_console(out)) {
            std::wstring w = widen(text);
            DWORD written = 0;
            WriteConsoleW(out, w.data(), static_cast<DWORD>(w.size()), &written, nullptr);
        }
        else {
            std::fwrite(text.data(), 1, text.size(), stdout);
            std::fflush(stdout);
        }
    }

    bool read_line(std::string& line) {
        HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
        if (!is_console(in)) {
            return static_cast<bool>(std::getline(std::cin, line));
        }
        // the console gives UTF-16, plain getline would break non ASCII input
        std::wstring w;
        wchar_t buffer[256];
        while (w.empty() || w.back() != L'\n') {
            DWORD read = 0;
            if (!ReadConsoleW(in, buffer, 256, &read, nullptr) || read == 0) {
                return false;
            }
            w.append(buffer, read);
        }
        while (!w.empty() && (w.back() == L'\n' || w.back() == L'\r')) {
            w.pop_back();
        }
        line = narrow(w);
        return true;
    }

    int width() {
        CONSOLE_SCREEN_BUFFER_INFO info;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)) {
            return info.srWindow.Right - info.srWindow.Left + 1;
        }
        return 80;
    }

#else

    void init() {}

    void write(const std::string& text) {
        std::fwrite(text.data(), 1, text.size(), stdout);
        std::fflush(stdout);
    }

    bool read_line(std::string& line) {
        return static_cast<bool>(std::getline(std::cin, line));
    }

    int width() {
        return 80;
    }

#endif

}