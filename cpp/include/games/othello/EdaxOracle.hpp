#pragma once

#include <core/BasicTypes.hpp>
#include <games/othello/Game.hpp>

#include <boost/process.hpp>

#include <mutex>
#include <string>
#include <vector>

namespace othello {

class EdaxOracle {
 public:
  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;

  EdaxOracle(bool verbose, bool deterministic_mode);
  ~EdaxOracle();

  core::action_t query(int depth, const State& state, const ActionMask& valid_actions);

 private:
  std::vector<std::string> line_buffer_;
  std::vector<std::string> tokens_;

  boost::process::ipstream out_;
  boost::process::opstream in_;
  boost::process::child* child_;

  mutable std::mutex mutex_;
  const bool verbose_;
  const bool deterministic_mode_;
};

}  // namespace othello
