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
    using kNumActionsPerMode = util::int_sequence<nim::kMaxStonesToTake>;
    static constexpr int kNumPlayers = nim::kNumPlayers;
    static constexpr int kMaxBranchingFactor = nim::kMaxStonesToTake;
  };

  struct MctsConfiguration : public core::MctsConfigurationBase {
    static constexpr float kOpeningLength = 3;
  };

/*
 * This structure encapsulates the state of the game using a single integer uint8_t.
 * (bits) to store various pieces of information:
 * - Bits 0-4: Number of stones (0-31)
 * - Bit 5: Current player (0 or 1)
 * - Bit 6: Player readiness (0 or 1)
 *
 * It provides methods to get and set these values.
 */
  struct State {
    State() : bits(0) {}
    auto operator<=>(const State& other) const = default;
    size_t hash() const { return bits; }

    int get_stones() const { return bits & 0b11111; }  // Bits 0-4
    void set_stones(int stones) { bits = (bits & ~0b11111) | (stones & 0b11111); }
    bool get_player() const { return (bits >> 5) & 1; }  // Bit 5
    void set_player(bool player) { bits = (bits & ~(1 << 5)) | (player << 5); }
    bool is_player_ready() const { return (bits >> 6) & 1; }  // Bit 6
    void set_player_ready(bool ready) { bits = (bits & ~(1 << 6)) | (ready << 6); }

   private:
    state_t bits;
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
    static bool is_chance_mode(core::action_mode_t mode) { return mode == 1; }
    static bool is_chance(const State& state) { return is_chance_mode(get_action_mode(state)); }
    static Types::PolicyTensor get_chance_dist(const State& state);
    static void apply_chance(StateHistory& history);
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
      ss << "[" << state.get_stones() << ", " << state.get_player() << "]";
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

      for (int i = 0; i < state->get_stones(); ++i) {
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

#include <inline/games/nim/Game.inl>