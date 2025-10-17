#pragma once

#include <boost/json.hpp>

namespace generic {

/*
 * VerboseDataBase is the base class for verbose data structures used to expose detailed MCTS
 * search statistics.
 */
struct VerboseDataBase {
  virtual ~VerboseDataBase() {}
  virtual boost::json::object to_json() const = 0;
  virtual void to_terminal_text() const = 0;

  bool initialized = false;
};

}  // namespace generic
