#pragma once

#include <boost/json.hpp>

namespace generic {

struct VerboseDataBase {
  virtual ~VerboseDataBase() {}
  virtual boost::json::object to_json() const = 0;
  virtual void to_terminal_text(std::ostream& ss, int n_rows_to_display) const = 0;

  bool initialized = false;
};

}  // namespace generic
