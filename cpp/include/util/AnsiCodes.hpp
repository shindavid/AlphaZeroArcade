#pragma once

#include "util/Rendering.hpp"

/*
 * ANSI codes.
 *
 * Each of these functions accepts an optional argument that is used when util::Rendering::mode() is
 * util::Rendering::kText (i.e., when the output is not a terminal).
 */
namespace ansi {

inline const char* kCircle(const char* s = nullptr) {
  return util::Rendering::mode() == util::Rendering::kTerminal ? "\u25CF" : s;
}

inline const char* kRectangle(const char* s = nullptr) {
  return util::Rendering::mode() == util::Rendering::kTerminal ? "\u2587" : s;
}

inline const char* kBlink(const char* s = nullptr) {
  return util::Rendering::mode() == util::Rendering::kTerminal ? "\033[5m" : s;
}

inline const char* kRed(const char* s = nullptr) {
  return util::Rendering::mode() == util::Rendering::kTerminal ? "\033[31m" : s;
}

inline const char* kYellow(const char* s = nullptr) {
  return util::Rendering::mode() == util::Rendering::kTerminal ? "\033[33m" : s;
}

inline const char* kGreen(const char* s = nullptr) {
  return util::Rendering::mode() == util::Rendering::kTerminal ? "\033[32m" : s;
}

inline const char* kBlue(const char* s = nullptr) {
  return util::Rendering::mode() == util::Rendering::kTerminal ? "\033[34m" : s;
}

inline const char* kWhite(const char* s = nullptr) {
  return util::Rendering::mode() == util::Rendering::kTerminal ? "\033[37m" : s;
}

inline const char* kReset(const char* s = nullptr) {
  return util::Rendering::mode() == util::Rendering::kTerminal ? "\033[00m" : s;
}

}  // namespace ansi
