#pragma once

#include <string>
#include <vector>

#include <boost/process.hpp>
#include <boost/program_options.hpp>

#include <common/AbstractPlayer.hpp>
#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <othello/Constants.hpp>
#include <othello/GameState.hpp>
#include <util/BoostUtil.hpp>

namespace othello {

/*
 * EdaxPlayer is a player that uses the edax engine to play Othello.
 *
 * See: https://github.com/okuhara/edax-reversi-AVX
 *
 * Currently, we interact with the edax engine in a clunky, inefficient way: we launch the edax process, submit
 * moves to it via stdin, and parse the stdout to get the engine's response. Later, we can try to avoid the I/O and
 * the text parsing by building and linking against an edax library directly.
 */
class EdaxPlayer : public Player {
public:
  using base_t = Player;

  struct Params {
    int depth = 21;  // matches edax default

    auto make_options_description();
  };

  EdaxPlayer(const Params&);

  void start_game() override;
  void receive_state_change(common::seat_index_t, const GameState&, common::action_index_t) override;
  common::action_index_t get_action(const GameState&, const ActionMask&) override;

private:
  const Params params_;
  std::vector<std::string> line_buffer_;
  boost::process::ipstream out_;
  boost::process::opstream in_;
  boost::process::child* proc_ = nullptr;
};

}  // namespace othello

#include <othello/players/inl/EdaxPlayer.inl>
