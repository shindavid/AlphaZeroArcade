#pragma once

# include "search/VerboseDataBase.hpp"

namespace generic {

struct VerboseManager {
  ~VerboseManager() { delete verbose_data_; }

  static VerboseManager* get_instance() {
    static VerboseManager instance;
    return &instance;
  }

  void set(VerboseDataBase* verbose_data) {
    if (!tui_player_registered_ && !web_player_registered_) {
      verbose_data_->to_terminal_text();
    } else {
      verbose_data_ = verbose_data;
    }
  }

  VerboseDataBase* verbose_data() const { return verbose_data_; }
  void register_tui_player() { tui_player_registered_ = true; }
  void register_web_player() { web_player_registered_ = true; }

  private:
    VerboseDataBase* verbose_data_ = nullptr;
    int num_rows_to_display_ = -1;
    bool tui_player_registered_ = false;
    bool web_player_registered_ = false;
};

}  // namespace generic
