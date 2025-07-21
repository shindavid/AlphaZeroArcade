#pragma once

#include "util/CppUtil.hpp"

/*
 * ANSI codes.
 */
namespace ansi {

/*
 * Optional argument is used in non-tty mode.
 */
inline const char* kCircle(const char* s=nullptr) { return util::tty_mode() ? "\u25CF" : s; }
inline const char* kRectangle(const char* s=nullptr) { return util::tty_mode() ? "\u2587" : s; }
inline const char* kBlink(const char* s=nullptr) { return util::tty_mode() ? "\033[5m" : s; }
inline const char* kRed(const char* s=nullptr) { return util::tty_mode() ? "\033[31m" : s; }
inline const char* kYellow(const char* s=nullptr) { return util::tty_mode() ? "\033[33m" : s; }
inline const char* kGreen(const char* s=nullptr) { return util::tty_mode() ? "\033[32m" : s; }
inline const char* kBlue(const char* s=nullptr) { return util::tty_mode() ? "\033[34m" : s; }
inline const char* kWhite(const char* s=nullptr) { return util::tty_mode() ? "\033[37m" : s; }
inline const char* kReset(const char* s=nullptr) { return util::tty_mode() ? "\033[00m" : s; }

}  // namespace ansi;
