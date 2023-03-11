#pragma once

/*
 * ANSI codes.
 */

namespace ansi {

class Codes {
public:
  static const char* kCircle() { return instance()->kCircle_; }
  static const char* kBlink() { return instance()->kBlink_; }
  static const char* kRed() { return instance()->kRed_; }
  static const char* kYellow() { return instance()->kYellow_; }
  static const char* kReset() { return instance()->kReset_; }

private:
  Codes();

  static Codes* instance();
  static Codes* instance_;

  const char* kCircle_ = nullptr;
  const char* kBlink_ = nullptr;
  const char* kRed_ = nullptr;
  const char* kYellow_ = nullptr;
  const char* kReset_ = nullptr;
};

inline const char* kCircle() { return Codes::kCircle(); }
inline const char* kBlink() { return Codes::kBlink(); }
inline const char* kRed() { return Codes::kRed(); }
inline const char* kYellow() { return Codes::kYellow(); }
inline const char* kReset() { return Codes::kReset(); }

}  // namespace ansi;

#include <util/inl/AnsiCodes.inl>
