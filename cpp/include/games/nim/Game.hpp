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
#include <games/nim/Constants.hpp>
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

namespace nim {

struct Game {
  struct Constants : public core::ConstantsBase {
    static constexpr int kNumPlayers = nim::kNumPlayers;
    static constexpr int kNumActions = nim::kMaxStonesToTake;
    static constexpr int kMaxBranchingFactor = nim::kMaxStonesToTake;
  };

  struct MctsConfiguration : public core::MctsConfigurationBase {
    static constexpr float kOpeningLength = 3;
  };

  struct State {
    auto operator<=>(const State& other) const = default;

    size_t hash() const {
      auto tuple = std::make_tuple(stones_left, current_player);
      std::hash<decltype(tuple)> hasher;
      return hasher(tuple);
    }

    int stones_left;
    int current_player;
  };

  using GameResults = core::WinShareResults<Constants::kNumPlayers>;
  using StateHistory = core::SimpleStateHistory<State, Constants::kNumPreviousStatesToEncode>;
  using SymmetryGroup = groups::TrivialGroup;
  using Symmetries = core::TrivialSymmetries;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Rules {
    static void init_state(State& state) {
      state.stones_left = nim::kStartingStones;
      state.current_player = 0;
    }

    static Types::ActionMask get_legal_moves(const StateHistory& history) {
      const State& state = history.current();
      Types::ActionMask mask;

      for (int i = 0; i < nim::kMaxStonesToTake; ++i) {
        mask[i] = i + 1 <= state.stones_left;
      }

      return mask;
    }

    static core::seat_index_t get_current_player(const State& state) {
      return state.current_player;
    }

    static void apply(StateHistory& history, core::action_t action) {
      if (action < 0 || action >= nim::kMaxStonesToTake) {
        throw std::invalid_argument("Invalid action: " + std::to_string(action));
      }

      State& state = history.extend();
      state.stones_left -= action + 1;
      state.current_player = 1 - state.current_player;
    }

    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome) {
      if (state.stones_left == 0) {
        outcome.setZero();
        outcome(last_player) = 1;
        return true;
      }
      return false;
    }
  };

  struct IO : public core::IOBase<Types, State> {
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action) { return std::to_string(action + 1); }
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
      ss << "[" << state.stones_left << ", " << state.current_player << "]";
      return ss.str();
    }
  };

  struct InputTensorizor {
    using Tensor = eigen_util::FTensor<Eigen::Sizes<nim::kStartingStones>>;
    using MCTSKey = State;
    using EvalKey = State;

    static MCTSKey mcts_key(const StateHistory& history) { return history.current(); }
    template <typename Iter>
    static EvalKey eval_key(Iter start, Iter cur) {
      return *cur;
    }
    template <typename Iter>
    static Tensor tensorize(Iter start, Iter cur) {
      Tensor tensor;
      tensor.setZero();
      Iter state = cur;

      for (int i = 0; i < state->stones_left; ++i) {
        tensor(nim::kStartingStones - 1 - i) = 1;
      }
      return tensor;
    }
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
}  // namespace nim

namespace std {

template <>
struct hash<nim::Game::State> {
  size_t operator()(const nim::Game::State& pos) const { return pos.hash(); }
};
}  // namespace std

static_assert(core::concepts::Game<nim::Game>);