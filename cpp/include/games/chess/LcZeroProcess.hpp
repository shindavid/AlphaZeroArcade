#pragma once

#include <boost/process.hpp>

namespace a0achess {

class LcZeroProcess {
 public:

  LcZeroProcess();
  ~LcZeroProcess();

  std::string query(const std::string& fen_move_str, const std::string& go_cmd);
  static std::string parse_bestmove_line(const std::string& line);

 private:
  boost::process::child* process_;
  boost::process::ipstream out_;
  boost::process::opstream in_;
};

}  // namespace a0achess
