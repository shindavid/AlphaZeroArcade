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
#include <util/AnsiCodes.hpp>
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
    static constexpr int kMaxBranchingFactor =
        std::max(stochastic_nim::kMaxStonesToTake, stochastic_nim::kChanceDistributionSize);
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
    static void print_state(std::ostream& ss, const State& state, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr) {
      ss << compact_state_repr(state) << std::endl;
    }
    static void print_mcts_results(std::ostream& ss, const Types::PolicyTensor& action_policy,
                                   const Types::SearchResults& results) {
      const auto& valid_actions = results.valid_actions;
      const auto& mcts_counts = results.counts;
      const auto& net_policy = results.policy_prior;
      const auto& win_rates = results.win_rates;
      const auto& net_value = results.value_prior;

      constexpr int buf_size = 4096;
      char buffer[buf_size];
      int cx = 0;

      cx += snprintf(buffer + cx, buf_size - cx, "%8s %6s%s%s%s %6s%s%s%s\n", "", "",
                     ansi::kRed(""), ansi::kCircle("R"), ansi::kReset(""), "", ansi::kYellow(""),
                     ansi::kCircle("Y"), ansi::kReset(""));
      cx += snprintf(buffer + cx, buf_size - cx, "%8s %6.3f%% %6.3f%%\n", "net(W)",
                     100 * net_value(0), 100 * net_value(1));
      cx += snprintf(buffer + cx, buf_size - cx, "%8s %6.3f%% %6.3f%%\n", "net(D)",
                     100 * net_value(2), 100 * net_value(2));
      cx += snprintf(buffer + cx, buf_size - cx, "%8s %6.3f%% %6.3f%%\n", "net(L)",
                     100 * net_value(1), 100 * net_value(0));
      cx += snprintf(buffer + cx, buf_size - cx, "%8s %6.3f%% %6.3f%%\n", "win-rate",
                     100 * win_rates(0), 100 * win_rates(1));
      cx += snprintf(buffer + cx, buf_size - cx, "\n");

      cx += snprintf(buffer + cx, buf_size - cx, "%3s %8s %8s %8s\n", "Col", "Net", "Count",
                     "Action");

      for (int i = 0; i < stochastic_nim::kMaxStonesToTake; ++i) {
        if (valid_actions[i]) {
          cx += snprintf(buffer + cx, buf_size - cx, "%3d %8.3f %8.3f %8.3f\n", i + 1,
                         net_policy(i), mcts_counts(i), action_policy(i));
        } else {
          cx += snprintf(buffer + cx, buf_size - cx, "%3d\n", i + 1);
        }
      }

      util::release_assert(cx < buf_size, "Buffer overflow (%d < %d)", cx, buf_size);
      ss << buffer << std::endl;
    }

    static std::string compact_state_repr(const State& state) {
      std::ostringstream ss;
      ss << "p" << state.current_player;
      if (state.current_mode == stochastic_nim::kChanceMode) {
        ss << "*";
      }
      ss << "@" << state.stones_left;
      return ss.str();
    }
  };

  struct InputTensorizor {
    // tensor is of the format {binary encoding of stones_left, current_player, current_mode}
    constexpr static int kNumFeatures = stochastic_nim::kStartingStonesBitWidth + 2;
    using Tensor = eigen_util::FTensor<Eigen::Sizes<1, kNumFeatures, 1>>;
    using MCTSKey = State;
    using EvalKey = State;

    static MCTSKey mcts_key(const StateHistory& history) { return history.current(); }
    template <typename Iter>
    static EvalKey eval_key(Iter start, Iter cur) { return *cur; }
    template <typename Iter>
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

