#pragma once

#include "search/VerboseDataBase.hpp"

#include <memory>

namespace generic {
/*
 * VerboseManager is a singleton that maintains a pointer to the current VerboseData instance.
 * After each MCTS search, a player can set this pointer to expose detailed search statistics
 * and diagnostics.
 *
 * By default, VerboseManager automatically prints the VerboseData to the terminal after every
 * completed search (auto_terminal_printing_ = true). Player generators that provide their own
 * output interfaces—such as WebPlayerGenerator or TUIPlayerGenerator—disable this behavior by
 * calling disable_auto_terminal_printing().
 */
struct VerboseManager {
  using VerboseDataBase_sptr = std::shared_ptr<VerboseDataBase>;

  static VerboseManager* get_instance();
  void set(VerboseDataBase_sptr verbose_data);

  VerboseDataBase_sptr verbose_data() const { return verbose_data_; }
  void disable_auto_terminal_printing() { auto_terminal_printing_enabled_ = false; }
  bool auto_terminal_printing_enabled() const { return auto_terminal_printing_enabled_; }

 private:
  VerboseDataBase_sptr verbose_data_ = nullptr;
  bool auto_terminal_printing_enabled_ = true;
};

}  // namespace generic

#include "inline/search/VerboseManager.inl"
