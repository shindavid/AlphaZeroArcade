#pragma once

#include "boost/json/object.hpp"

namespace core {
struct WebManagerClient {
  virtual void handle_action(const boost::json::object& payload) = 0;
  virtual void handle_resign() = 0;
};
}  // namespace core
