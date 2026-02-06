#pragma once

#include "core/BasicTypes.hpp"
#include "core/ConstantsBase.hpp"
#include "core/GameRulesBase.hpp"
#include "core/GameTypes.hpp"
#include "core/IOBase.hpp"
#include "core/TrivialSymmetries.hpp"
#include "core/WinShareResults.hpp"
#include "core/concepts/GameConcept.hpp"
#include "games/stochastic_nim/Constants.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/functional/hash.hpp>

#include <functional>
#include <string>

namespace stochastic_nim {

struct Game {
  struct Constants : public core::ConstantsBase {
    static constexpr const char* kGameName = "stochastic_nim";
    using kNumActionsPerMode =
      util::int_sequence<stochastic_nim::kMaxStonesToTake, stochastic_nim::kChanceDistributionSize>;
    static constexpr int kNumPlayers = stochastic_nim::kNumPlayers;
    static constexpr int kMaxBranchingFactor =
      std::max(stochastic_nim::kMaxStonesToTake, stochastic_nim::kChanceDistributionSize);
  };

  struct State {
    auto operator<=>(const State& other) const = default;
    size_t hash() const;

    int stones_left = 0;
    int current_player = 0;
    core::action_mode_t current_mode = 0;
  };

  using GameResults = core::WinShareResults<Constants::kNumPlayers>;
  using SymmetryGroup = groups::TrivialGroup;
  using Symmetries = core::TrivialSymmetries;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Rules : public core::RulesBase<Types> {
    static void init_state(State& state);
    static Types::ActionMask get_legal_moves(const State& state);
    static core::action_mode_t get_action_mode(const State& state);
    static core::seat_index_t get_current_player(const State& state);
    static void apply(State& state, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);
    static bool is_chance_mode(const core::action_mode_t& mode);
    static Types::ChanceDistribution get_chance_distribution(const State& state);
  };

  struct IO : public core::IOBase<Types> {
    static constexpr char kSeatChars[Constants::kNumPlayers] = {'A', 'B'};
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action, core::action_mode_t mode);
    static void print_state(std::ostream& ss, const State& state, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);
    static std::string compact_state_repr(const State& state);
  };

  static void static_init() {}
};  // struct Game
}  // namespace stochastic_nim

namespace std {

template <>
struct hash<stochastic_nim::Game::State> {
  size_t operator()(const stochastic_nim::Game::State& pos) const { return pos.hash(); }
};
}  // namespace std

static_assert(core::concepts::Game<stochastic_nim::Game>);

#include "inline/games/stochastic_nim/Game.inl"

// Ensure that we always have bindings when we #include "games/stochastic_nim/Game.hpp":
#include "games/stochastic_nim/Bindings.hpp"  // IWYU pragma: keep
