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
    using kNumActionsPerMode = util::int_sequence<nim::kMaxStonesToTake, nim::kMaxRandomStonesToTake>;
    static constexpr int kNumPlayers = nim::kNumPlayers;
    static constexpr int kMaxBranchingFactor = nim::kMaxStonesToTake;
  };

  struct MctsConfiguration : public core::MctsConfigurationBase {
    static constexpr float kOpeningLength = 3;
  };

  struct State {
    auto operator<=>(const State& other) const = default;
    size_t hash() const;

    int get_stones() const { return stones_left; }  // Bits 0-4
    void set_stones(int stones) { stones_left = stones; }
    bool get_player() const { return current_player; }  // Bit 5
    void set_player(bool player) { current_player = player; }
    bool is_player_ready() const { return player_ready; }  // Bit 6
    void set_player_ready(bool ready) { player_ready = ready; }

   private:
    int stones_left;
    bool current_player;
    bool player_ready;
  };

  using GameResults = core::WinShareResults<Constants::kNumPlayers>;
  using StateHistory = core::SimpleStateHistory<State, Constants::kNumPreviousStatesToEncode>;
  using SymmetryGroup = groups::TrivialGroup;
  using Symmetries = core::TrivialSymmetries;
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;

  struct Rules {
    static void init_state(State& state);
    static Types::ActionMask get_legal_moves(const StateHistory& history);
    // action mode: 0 means player's move, 1 means chance move
    static core::action_mode_t get_action_mode(const State& state) { return !state.is_player_ready(); }
    static core::seat_index_t get_current_player(const State& state) { return state.get_player(); }
    static void apply(StateHistory& history, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);
    static int get_num_chance_actions() { return kMaxRandomStonesToTake + 1; }
    static bool has_known_dist(const State& state) { return (get_action_mode(state) == 1); }
    static Types::PolicyTensor get_known_dist(const State& state);
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
      ss << "[" << state.get_stones() << ", " << state.get_player() << ", "
         << state.is_player_ready() << "]";
      return ss.str();
    }
  };

  struct InputTensorizor {
    using Tensor = eigen_util::FTensor<Eigen::Sizes<3>>;
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

      tensor(0) = state->get_stones();
      tensor(1) = state->get_player();
      tensor(2) = state->is_player_ready();
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

#include <inline/games/nim/Game.inl>