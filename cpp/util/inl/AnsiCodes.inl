#include <util/AnsiCodes.hpp>

#include <util/CppUtil.hpp>

namespace ansi {

inline Codes::Codes() {
  if (util::tty_mode()) {
    kCircle_ = "\u25CF";
    kBlink_ = "\033[5m";
    kRed_ = "\033[31m";
    kYellow_ = "\033[33m";
    kReset_ = "\033[00m";
  } else {
    kCircle_ = "";
    kBlink_ = "";
    kRed_ = "R";
    kYellow_ = "Y";
    kReset_ = "";
  }
}

inline Codes* Codes::instance() {
  if (!instance_) {
    instance_ = new Codes();
  }
  return instance_;
}

}  // namespace ansi
