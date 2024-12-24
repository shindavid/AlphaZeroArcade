#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/ConstantsBase.hpp>
#include <core/GameLog.hpp>
#include <core/GameTypes.hpp>
#include <core/MctsConfigurationBase.hpp>
#include <core/IOBase.hpp>
#include <core/SimpleStateHistory.hpp>
#include <core/TrainingTargets.hpp>
#include <core/TrivialSymmetries.hpp>
#include <core/WinShareResults.hpp>
#include <games/stochastic_nim/Constants.hpp>
#include <games/GameRulesBase.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>
#include <util/MetaProgramming.hpp>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>

namespace stochastic_nim {

struct Game {
  struct Constants : public core::ConstantsBase {
    using kNumActionsPerMode =
        util::int_sequence<stochastic_nim::kMaxStonesToTake, stochastic_nim::kChanceDistributionSize>;
    static constexpr int kNumPlayers = stochastic_nim::kNumPlayers;
    static constexpr int kMaxBranchingFactor = stochastic_nim::kMaxStonesToTake;
  };

  struct MctsConfiguration : public core::MctsConfigurationBase {
    static constexpr float kOpeningLength = 3;
  };

  struct State {
    auto operator<=>(const State& other) const = default;
    size_t hash() const;

    int stones_left;
    int current_player;
    core::action_mode_t current_mode;
  };

  using GameResults = core::WinShareResults<Constants::kNumPlayers>;
  using StateHistory = core::SimpleStateHistory<State, Constants::kNumPreviousStatesToEncode>;
  using SymmetryGroup = groups::TrivialGroup;
  using Symmetries = core::TrivialSymmetries;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Rules : public game_base::RulesBase<Types> {
    static void init_state(State& state);
    static Types::ActionMask get_legal_moves(const StateHistory& history);
    static core::action_mode_t get_action_mode(const State& state) { return state.current_mode; }
    static core::seat_index_t get_current_player(const State& state) { return state.current_player; }
    static void apply(StateHistory& history, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);
    static bool is_chance_mode(const core::action_mode_t& mode) {
      return mode == stochastic_nim::kChanceMode;
    }
    static Types::ChanceDistribution get_chance_distribution(const State& state);
  };

  struct IO : public core::IOBase<Types, State> {
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action, core::action_mode_t) {
      return std::to_string(action + 1);
    }
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr) {
      throw std::runtime_error("Not implemented");
    }
    static void print_mcts_results(std::ostream&, const Types::PolicyTensor& action_policy,
                                   const Types::SearchResults&) {
      throw std::runtime_error("Not implemented");
    }
    static std::string compact_state_repr(const State& state) {
      std::ostringstream ss;
      ss << "p" << state.current_player;
      if (state.current_mode == stochastic_nim::kChanceMode) {
        ss << "*";
      }
      ss << " @" << state.stones_left;
      return ss.str();
    }
  };

  struct InputTensorizor {
    using Tensor = eigen_util::FTensor<Eigen::Sizes<stochastic_nim::kStartingStonesBitWidth + 2>>;
    using MCTSKey = State;
    using EvalKey = State;

    static MCTSKey mcts_key(const StateHistory& history) { return history.current(); }
    template <typename Iter>
    static EvalKey eval_key(Iter start, Iter cur) { return *cur; }
    template <typename Iter>
    // tensor is of the format {binary encoding of stones_left, current_player, current_mode}
    static Tensor tensorize(Iter start, Iter cur);
  };

  struct TrainingTargets {
    using PolicyTarget = core::PolicyTarget<Game>;
    using ValueTarget = core::ValueTarget<Game>;
    using ActionValueTarget = core::ActionValueTarget<Game>;
    using OppPolicyTarget = core::OppPolicyTarget<Game>;
    using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget>;
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

#include <inline/games/stochastic_nim/Game.inl>

