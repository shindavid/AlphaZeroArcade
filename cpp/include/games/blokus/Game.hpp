#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/TrivialSymmetries.hpp"
#include "core/WinShareResults.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/blokus/Constants.hpp"
#include "games/blokus/GameState.hpp"
#include "util/CppUtil.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/functional/hash.hpp>

#include <functional>
#include <string>

namespace blokus {

class Game {
 public:
  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "blokus";
    using kNumActionsPerMode = util::int_sequence<kNumLocationActions, kNumPiecePlacementActions>;
    static constexpr int kNumPlayers = blokus::kNumPlayers;
    static constexpr int kMaxBranchingFactor = blokus::kNumPieceOrientationCorners;
    static constexpr char kSeatChars[kNumPlayers] = {'B', 'Y', 'R', 'G'};
  };

  using State = blokus::GameState;
  using GameResults = core::WinShareResults<Constants::kNumPlayers>;

  /*
   * After the initial placement of the first piece, the rules of the game are symmetric. But the
   * rules are not symmetric for the first piece placement, and as a result, strategic
   * considerations are asymmetric for much, if not all of the game. Because of this, it's unclear
   * whether exploiting symmetry will be useful, so we use the trivial group.
   */
  using SymmetryGroup = groups::TrivialGroup;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;
  using Symmetries = core::TrivialSymmetries;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State&);
    static Types::ActionMask get_legal_moves(const State&);
    static core::action_mode_t get_action_mode(const State&);
    static core::seat_index_t get_current_player(const State&);
    static void apply(State&, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);

   private:
    static GameResults::Tensor compute_outcome(const State& state);
  };

  struct IO : public core::IOBase<Types> {
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action, core::action_mode_t);
    static std::string player_to_str(core::seat_index_t player);
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);

    /*
     * Inverse operation of print_state(ss, state) in non-tty-mode.
     *
     * Assumes that the last pass_count players have passed.
     */
    static State load(const std::string& str, int pass_count = 0);
  };

  static void static_init() {}
};

}  // namespace blokus

namespace std {

template <>
struct hash<blokus::Game::State> {
  size_t operator()(const blokus::Game::State& pos) const { return pos.hash(); }
};

}  // namespace std

static_assert(core::concepts::Game<blokus::Game>);

#include "inline/games/blokus/Game.inl"

// Ensure that we always have bindings when we #include "games/blokus/Game.hpp":
#include "games/blokus/Bindings.hpp"  // IWYU pragma: keep
