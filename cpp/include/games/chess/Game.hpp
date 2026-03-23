#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/WinLossDrawResults.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/chess/Constants.hpp"
#include "games/chess/GameState.hpp"
#include "games/chess/InputFrame.hpp"
#include "util/CppUtil.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/functional/hash.hpp>

#include <string>

namespace a0achess {

struct Game {
  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "chess";
    using kNumActionsPerMode = util::int_sequence<a0achess::kNumActions>;
    static constexpr int kNumPlayers = a0achess::kNumPlayers;
    static constexpr int kMaxBranchingFactor = a0achess::kMaxBranchingFactor;

    // 2 = official rules, 1 = engine rules
    static constexpr int kRepetitionDrawThreshold = 1;
  };

  using State = GameState;
  using GameResults = core::WinLossDrawResults;
  using SymmetryGroup = groups::TrivialGroup;  // TODO: Implement symmetries
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State&);
    static core::action_mode_t get_action_mode(const State&) { return 0; }
    static core::seat_index_t get_current_player(const State&);
    static void apply(State&, core::action_t action);
    static void backtrack_state(State& state, const State& prev_state) {
      state.backtrack_to(prev_state);
    }
    static Result analyze(const State& state);
    static Result analyze(const InputFrame&);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[Constants::kNumPlayers] = {'W', 'B'};
    static std::string action_delimiter() { return ""; }
    static std::string action_to_str(core::action_t action, core::action_mode_t);
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);
  };

  static void static_init() {}
};

}  // namespace a0achess

static_assert(core::concepts::Game<a0achess::Game>);

#include "inline/games/chess/Game.inl"

// Ensure that we always have bindings when we #include "games/chess/Game.hpp":
#include "games/chess/Bindings.hpp"  // IWYU pragma: keep
