#pragma once

#include <cstdint>
#include <unistd.h>
#include <vector>

namespace util {

/**
 * @brief Stack-based rendering mode context for text vs. terminal output.
 *
 * Rendering::get() returns the current rendering mode, which by default is determined by
 * isatty(STDOUT_FILENO): kTerminal if true, kText otherwise. This allows code to branch
 * on rendering mode (e.g., colored output for terminal, plain text otherwise).
 *
 * The mode can be temporarily overridden using push()/pop() or the RAII Guard.
 *
 * Example usage:
 *   if (util::Rendering::get() == util::Rendering::kTerminal) { ... }
 *   {
 *     util::Rendering::Guard g(util::Rendering::kText); // force text mode in this scope
 *     ...
 *   }
 */
class Rendering {
 public:
  /**
   * @brief Rendering mode: kText for plain text, kTerminal for terminal (TTY) output.
   */
  enum Mode : int8_t {
    kText,     ///< Plain text rendering mode
    kTerminal  ///< Terminal (TTY) rendering mode (e.g., with color/graphics)
  };

  /**
   * @brief RAII guard for temporarily setting rendering mode (restores previous mode on
   * destruction).
   * Usage: Rendering::Guard g(Rendering::kText);
   */
  struct Guard {
    Guard(Mode mode);
    ~Guard();
  };

  /**
   * @brief Get the current rendering mode (top of the mode stack).
   */
  static Mode mode();

  /**
   * @brief Set the base rendering mode (bottom of the stack).
   * This is rarely needed; prefer push()/pop() or Guard for scoped overrides.
   */
  static void set(Mode mode);

  /**
   * @brief Push a new rendering mode onto the stack (overrides current mode).
   */
  static void push(Mode mode);

  /**
   * @brief Pop the top rendering mode from the stack.
   * @throws util::CleanException if this would leave the stack empty.
   */
  static void pop();

 private:
  Rendering();
  static Rendering& instance();

  using vec_t = std::vector<Mode>;
  vec_t mode_stack_;
};

}  // namespace util

#include "inline/util/Rendering.inl"
