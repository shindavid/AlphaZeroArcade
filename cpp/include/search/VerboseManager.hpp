#pragma once

# include "search/VerboseDataBase.hpp"

namespace generic {

struct VerboseManager {
  static VerboseManager* get_instance();
  void set(VerboseDataBase* verbose_data);

  VerboseDataBase* verbose_data() const { return verbose_data_; }
  void disable_auto_terminal_printing() { auto_terminal_printing_enabled_ = false; }
  bool auto_terminal_printing_enabled() const { return auto_terminal_printing_enabled_; }

 private:
  VerboseDataBase* verbose_data_ = nullptr;
  int num_rows_to_display_ = -1;
  bool auto_terminal_printing_enabled_ = true;
};

}  // namespace generic

#include "inline/search/VerboseManager.inl"