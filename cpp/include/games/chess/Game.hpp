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
#include <core/TrainingTargets.hpp>
#include <games/chess/Constants.hpp>
#include <games/chess/LcZeroPositionHistoryAdapter.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>
#include <util/MetaProgramming.hpp>

#include <lc0/chess/position.h>

namespace chess {

struct Game {
  struct Constants {
    static constexpr int kNumPlayers = chess::kNumPlayers;
    static constexpr int kNumActions = chess::kNumActions;
    static constexpr int kMaxBranchingFactor = chess::kMaxBranchingFactor;
    static constexpr int kNumPreviousStatesToEncode = chess::kNumPreviousStatesToEncode;
    static constexpr float kOpeningLength = 18;  // 9 moves per player = reasonablish quarter-life
  };

  using State = lczero::Position;
  using StateHistory = chess::LcZeroPositionHistoryAdapter;
  using SymmetryGroup = groups::TrivialGroup;  // TODO: Implement symmetries
  using Types = core::GameTypes<Constants, State, SymmetryGroup>;

  struct Symmetries {
    static Types::SymmetryMask get_mask(const State& state);
    static void apply(State& state, group::element_t sym) {}
    static void apply(Types::PolicyTensor& policy, group::element_t sym) {}
    static void apply(core::action_t& action, group::element_t sym) {}
    static group::element_t get_canonical_symmetry(const State& state) { return 0; }
  };

  struct Rules {
    static void init_state(State&);
    static Types::ActionMask get_legal_moves(const StateHistory&);
    static core::seat_index_t get_current_player(const State&);
    static Types::ActionOutcome apply(StateHistory&, core::action_t action);
  };

  struct IO {
    static std::string action_delimiter() { return ""; }
    static std::string action_to_str(core::action_t action);
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr);
    static void print_mcts_results(std::ostream&, const Types::PolicyTensor& action_policy,
                                   const Types::SearchResults&);
  };

  struct InputTensorizor {
    static constexpr int kDim0 = kNumPlayers * (1 + Constants::kNumPreviousStatesToEncode);
    using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDim, kBoardDim>>;
    using MCTSKey = uint64_t;
    using EvalKey = State;

    static MCTSKey mcts_key(const StateHistory& history);
    template <typename Iter> static EvalKey eval_key(Iter start, Iter cur) { return *cur; }
    template <typename Iter> static Tensor tensorize(Iter start, Iter cur);
  };

  struct TrainingTargets {
    using BoardShape = Eigen::Sizes<kBoardDim, kBoardDim>;

    using PolicyTarget = core::PolicyTarget<Game>;
    using ValueTarget = core::ValueTarget<Game>;
    using ActionValueTarget = core::ActionValueTarget<Game>;
    using OppPolicyTarget = core::OppPolicyTarget<Game>;

    using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget>;
  };

  static void static_init() { lczero::InitializeMagicBitboards(); }
};

}  // namespace chess

namespace std {

template <>
struct hash<lczero::Position> {
  size_t operator()(const lczero::Position& pos) const { return pos.Hash(); }
};

}  // namespace std

static_assert(core::concepts::Game<chess::Game>);

#include <inline/games/chess/Game.inl>