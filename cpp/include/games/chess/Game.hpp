#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <core/ConstantsBase.hpp>
#include <core/GameLog.hpp>
#include <core/GameTypes.hpp>
#include <core/MctsConfigurationBase.hpp>
#include <core/IOBase.hpp>
#include <core/TrainingTargets.hpp>
#include <core/TrivialSymmetries.hpp>
#include <core/WinLossDrawResults.hpp>
#include <games/chess/Constants.hpp>
#include <games/chess/LcZeroPositionHistoryAdapter.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>
#include <util/MetaProgramming.hpp>

#include <lc0/chess/position.h>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>

namespace chess {

struct Game {
  struct Constants : public core::ConstantsBase {
    using kNumActionsPerMode = util::int_sequence<chess::kNumActions>;
    static constexpr int kNumPlayers = chess::kNumPlayers;
    static constexpr int kMaxBranchingFactor = chess::kMaxBranchingFactor;
    static constexpr int kNumPreviousStatesToEncode = chess::kNumPreviousStatesToEncode;
  };

  struct MctsConfiguration : public core::MctsConfigurationBase {
    static constexpr float kOpeningLength = 18;  // 9 moves per player = reasonablish quarter-life
  };

  using State = lczero::Position;
  using GameResults = core::WinLossDrawResults;
  using StateHistory = chess::LcZeroPositionHistoryAdapter;
  using SymmetryGroup = groups::TrivialGroup;  // TODO: Implement symmetries
  using Types = core::GameTypes<Constants, State, GameResults, SymmetryGroup>;
  using Symmetries = core::TrivialSymmetries;

  struct Rules {
    static void init_state(State&);
    static Types::ActionMask get_legal_moves(const StateHistory&);
    static core::action_mode_t get_action_mode(const State&) { return 0; }
    static core::seat_index_t get_current_player(const State&);
    static void apply(StateHistory&, core::action_t action);
    static bool is_terminal(const State& state, core::seat_index_t last_player,
                            core::action_t last_action, GameResults::Tensor& outcome);
  };

  struct IO : public core::IOBase<Types, State> {
    static std::string action_delimiter() { return ""; }
    static std::string action_to_str(core::action_t action, core::action_mode_t);
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
