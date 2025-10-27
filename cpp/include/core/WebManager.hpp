#pragma once

namespace core {


struct WebManager {
  static WebManager* get_instance();
};

}  // namespace core

#include "inline/core/WebManager.inl"
