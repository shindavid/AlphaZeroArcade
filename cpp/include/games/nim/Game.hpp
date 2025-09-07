#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/MctsConfigurationBase.hpp"
#include "core/SimpleStateHistory.hpp"
#include "core/TrainingTargets.hpp"
#include "core/TrivialSymmetries.hpp"
#include "core/WinShareResults.hpp"
#include "core/concepts/Game.hpp"
#include "games/GameRulesBase.hpp"
#include "games/nim/Constants.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"
#include "util/MetaProgramming.hpp"

#include <boost/functional/hash.hpp>

#include <functional>
#include <sstream>
#include <string>

namespace nim {

struct Game {
  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "nim";
    using kNumActionsPerMode = util::int_sequence<nim::kMaxStonesToTake>;
    static constexpr int kNumPlayers = nim::kNumPlayers;
    static constexpr int kMaxBranchingFactor = nim::kMaxStonesToTake;
  };

  struct MctsConfiguration : public core::MctsConfigurationBase {
    static constexpr float kOpeningLength = 3;
  };

  struct State {
    auto operator<=>(const State& other) const = default;
    size_t hash() const;

    int stones_left;
    int current_player;
  };

  using GameResults = core::WinShareResults<Constants::kNumPlayers>;
  using StateHistory = core::SimpleStateHistory<State, Constants::kNumPreviousStatesToEncode>;
  using SymmetryGroup = groups::TrivialGroup;
  using Symmetries = core::TrivialSymmetries;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Rules : public game_base::RulesBase<Types> {
    static void init_state(State& state);
    static Types::ActionMask get_legal_moves(const StateHistory& history);
    static core::action_mode_t get_action_mode(const State&) { return 0; }
    static core::seat_index_t get_current_player(const State& state);
    static void apply(StateHistory& history, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);
  };

  struct IO : public core::IOBase<Types> {
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action, core::action_mode_t);
    static void print_state(std::ostream& os, const State& state, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);
    static std::string compact_state_repr(const State& state);
  };

  static void static_init() {}
};  // struct Game

}  // namespace nim

namespace std {

template <>
struct hash<nim::Game::State> {
  size_t operator()(const nim::Game::State& pos) const { return pos.hash(); }
};

}  // namespace std

static_assert(core::concepts::Game<nim::Game>);

#include "inline/games/nim/Game.inl"  // IWYU pragma: keep
                                           //
// Ensure that we always have bindings when we #include "games/nim/Game.hpp":
#include "games/nim/Bindings.hpp"  // IWYU pragma: keep
