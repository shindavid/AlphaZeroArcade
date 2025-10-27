#include "core/WebManager.hpp"

namespace core {

inline WebManager* WebManager::get_instance() {
  static WebManager instance;
  return &instance;
}

}  // namespace core
