#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/TrivialSymmetries.hpp"
#include "core/WinLossDrawResults.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/chess/Constants.hpp"
#include "games/chess/GameState.hpp"
#include "lc0/chess/position.h"
#include "util/CppUtil.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/functional/hash.hpp>

#include <functional>
#include <string>

namespace chess {

struct Game {
  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "chess";
    using kNumActionsPerMode = util::int_sequence<chess::kNumActions>;
    static constexpr int kNumPlayers = chess::kNumPlayers;
    static constexpr int kMaxBranchingFactor = chess::kMaxBranchingFactor;
  };

  using State = GameState;
  using GameResults = core::WinLossDrawResults;
  using SymmetryGroup = groups::TrivialGroup;  // TODO: Implement symmetries
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;
  using Symmetries = core::TrivialSymmetries;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State&);
    static Types::ActionMask get_legal_moves(const State&);
    static core::action_mode_t get_action_mode(const State&) { return 0; }
    static core::seat_index_t get_current_player(const State&);
    static void apply(State&, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[Constants::kNumPlayers] = {'W', 'B'};
    static std::string action_delimiter() { return ""; }
    static std::string action_to_str(core::action_t action, core::action_mode_t);
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);
  };

  static void static_init() { lczero::InitializeMagicBitboards(); }
};

}  // namespace chess

namespace std {

template <>
struct hash<chess::Game::State> {
  size_t operator()(const chess::Game::State& state) const { return state.hash(); }
};

}  // namespace std

static_assert(core::concepts::Game<chess::Game>);

#include "inline/games/chess/Game.inl"

// Ensure that we always have bindings when we #include "games/chess/Game.hpp":
#include "games/chess/Bindings.hpp"  // IWYU pragma: keep
