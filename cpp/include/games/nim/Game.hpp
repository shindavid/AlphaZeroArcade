
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/GameLog.hpp>
#include <core/GameTypes.hpp>
#include <core/SimpleStateHistory.hpp>
#include <core/TrainingTargets.hpp>
#include <core/WinLossDrawResults.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>
#include <util/MetaProgramming.hpp>
#include <games/nim/Constants.hpp>
#include <core/TrivialSymmetries.hpp>

namespace nim {

struct Game {
  struct Constants {
    static constexpr int kNumPlayers = nim::kNumPlayers;
    static constexpr int kNumActions = nim::kMaxStonesToTake;
    static constexpr int kMaxBranchingFactor = nim::kMaxStonesToTake;
    static constexpr int kNumPreviousStatesToEncode = 0;
    static constexpr float kOpeningLength = 0.1;  // not applicable to Nim
  };

  struct State {
    auto operator<=>(const State& other) const = default;

    size_t hash() const {
      auto tuple = std::make_tuple(stones_left, num_moves_played);
      std::hash<decltype(tuple)> hasher;
      return hasher(tuple);
    }

    int stones_left;
    int num_moves_played;
  };

  using GameResults = core::WinLossDrawResults;
  using StateHistory = core::SimpleStateHistory<State, Constants::kNumPreviousStatesToEncode>;
  using SymmetryGroup = groups::TrivialGroup;
  using Symmetries = core::TrivialSymmetries;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Rules {
    static void init_state(State& state) {
      state.stones_left = nim::kStartingStones;
      state.num_moves_played = 0;
    }

    static Types::ActionMask get_legal_moves(const StateHistory& history) {
      const State& state = history.current();
      Types::ActionMask mask;

      for (int i = 1; i <= nim::kMaxStonesToTake; ++i) {
        mask[i] = i <= state.stones_left;
      }

      return mask;
    }

    static core::seat_index_t get_current_player(const State& state) {
      return state.num_moves_played % 2;
    }

    static Types::ActionOutcome apply(StateHistory& history, core::action_t action) {
      if (action < 1 || action > nim::kMaxStonesToTake) {
        throw std::invalid_argument("Invalid action: " + std::to_string(action));
      }


      State& state = history.current();
      state.stones_left -= action;
      state.num_moves_played++;

      if (state.stones_left == 0) {
        return GameResults::win(1 - get_current_player(state));
      }
      return Types::ActionOutcome();
    }
  };

  struct IO {
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action) { return std::to_string(action); }
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr) {
                              throw std::runtime_error("Not implemented");
                            }
    static void print_mcts_results(std::ostream&, const Types::PolicyTensor& action_policy,
                                   const Types::SearchResults&) {
                                      throw std::runtime_error("Not implemented");
                                   }
  };

  struct InputTensorizor {
    using Tensor = eigen_util::FTensor<Eigen::Sizes<2>>;
    using MCTSKey = State;
    using EvalKey = State;

    static MCTSKey mcts_key(const StateHistory& history) { return history.current(); }
    template <typename Iter> static EvalKey eval_key(Iter start, Iter cur) { return *cur; }
    template <typename Iter> static Tensor tensorize(Iter start, Iter cur) {
      Tensor tensor;
      tensor.setZero();
      Iter state = cur;
      tensor(0) = state.stones_left;
      tensor(1) = state.num_moves_played;
      return tensor;
    }
  };

  struct TrainingTargets {
    using BoardShape = Eigen::Sizes<1>;

    using PolicyTarget = core::PolicyTarget<Game>;
    using ValueTarget = core::ValueTarget<Game>;
    using ActionValueTarget = core::ActionValueTarget<Game>;
    using OppPolicyTarget = core::OppPolicyTarget<Game>;

    using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget>;
  };

  static void static_init() {}
}; // struct Game
}  // namespace nim

namespace std {

template <>
struct hash<nim::Game::State> {
  size_t operator()(const nim::Game::State& pos) const { return pos.hash(); }
};
}  // namespace std

static_assert(core::concepts::Game<nim::Game>);