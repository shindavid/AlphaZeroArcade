#pragma once

# include "search/VerboseDataBase.hpp"

namespace generic {

struct VerboseManager {
  static VerboseManager* get_instance() {
    static VerboseManager instance;
    return &instance;
  }

  void set(VerboseDataBase* verbose_data) {
    if (auto_terminal_printing_enabled_) {
      verbose_data_->to_terminal_text();
    } else {
      verbose_data_ = verbose_data;
    }
  }

  VerboseDataBase* verbose_data() const { return verbose_data_; }
  void disable_auto_terminal_printing() { auto_terminal_printing_enabled_ = false; }

 private:
  VerboseDataBase* verbose_data_ = nullptr;
  int num_rows_to_display_ = -1;
  bool auto_terminal_printing_enabled_ = true;
};

}  // namespace generic
