#include "search/VerboseManager.hpp"

namespace generic {

  inline VerboseManager* VerboseManager::get_instance() {
    static VerboseManager instance;
    return &instance;
  }

  void VerboseManager::set(VerboseDataBase* verbose_data) {
    if (auto_terminal_printing_enabled_) {
      verbose_data_->to_terminal_text();
    } else {
      verbose_data_ = verbose_data;
    }
  }

}  // namespace generic