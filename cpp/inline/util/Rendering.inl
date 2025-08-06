#include "util/Rendering.hpp"

#include "util/Exceptions.hpp"

namespace util {

inline Rendering::Guard::Guard(Mode mode) { push(mode); }

inline Rendering::Guard::~Guard() { pop(); }

inline Rendering::Mode Rendering::mode() {
  return instance().mode_stack_.back();
}

inline void Rendering::set(Mode mode) { instance().mode_stack_.front() = mode; }

inline void Rendering::push(Mode mode) {
  instance().mode_stack_.push_back(mode);
}

inline void Rendering::pop() {
  if (instance().mode_stack_.size() <= 1) {
    throw util::CleanException("Text mode stack underflow");
  }
  instance().mode_stack_.pop_back();
}

inline Rendering::Rendering() {
  Mode default_mode = isatty(STDOUT_FILENO) ? kTerminal : kText;
  mode_stack_.push_back(default_mode);
}

inline Rendering& Rendering::instance() {
  static Rendering instance;
  return instance;
}

}  // namespace util
