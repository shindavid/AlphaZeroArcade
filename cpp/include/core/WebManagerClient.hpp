#pragma once

#include "boost/json/object.hpp"

namespace core {
struct WebManagerClient {
  virtual void set_action(const boost::json::object& payload) = 0;
  virtual void set_resign() = 0;
};
}  // namespace core
