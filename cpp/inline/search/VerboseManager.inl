#include "search/VerboseManager.hpp"

namespace generic {

inline VerboseManager* VerboseManager::get_instance() {
  static VerboseManager instance;
  return &instance;
}

void VerboseManager::set(VerboseDataBase_sptr verbose_data) {
  if (verbose_data == nullptr) {
    return;
  }
  if (auto_terminal_printing_enabled_) {
    verbose_data->to_terminal_text();
  } else {
    // TODO: Support multi-player games by storing multiple VerboseData objects
    verbose_data_ = verbose_data;
  }
}

}  // namespace generic
