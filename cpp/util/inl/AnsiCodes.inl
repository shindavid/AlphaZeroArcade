#include <util/AnsiCodes.hpp>

#include <unistd.h>

namespace ansi {

inline Codes::Codes() {
  if (isatty(STDOUT_FILENO)) {  // https://stackoverflow.com/a/5157076/543913
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
