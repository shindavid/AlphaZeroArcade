#pragma once

#include <array>
#include <string>

#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/Types.hpp>

namespace common {

template<GameStateConcept GameState>
class AbstractPlayer {
public:
  using Result = typename common::GameStateTypes<GameState>::Result;
  using ActionMask = util::BitSet<GameState::kNumGlobalActions>;
  using player_array_t = std::array<AbstractPlayer *, GameState::kNumPlayers>;

  AbstractPlayer(const std::string &name) : name_(name) {}
  virtual ~AbstractPlayer() = default;
  void set_name(const std::string &name) { name_ = name; }
  std::string get_name() const { return name_; }

  virtual void start_game(const player_array_t &players, player_index_t seat_assignment) {}
  virtual void receive_state_change(player_index_t, const GameState &, action_index_t, const Result &) {}
  virtual action_index_t get_action(const GameState &, const ActionMask &) = 0;

private:
  std::string name_;
};

}  // namespace common
