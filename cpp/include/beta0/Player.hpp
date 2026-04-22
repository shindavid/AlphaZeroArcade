#pragma once

#include "beta0/Manager.hpp"
#include "beta0/SearchResults.hpp"
#include "beta0/concepts/SpecConcept.hpp"
#include "core/AbstractPlayer.hpp"
#include "core/ActionPrinter.hpp"
#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "core/Constants.hpp"
#include "core/InfoSetIterator.hpp"
#include "core/StateChangeUpdate.hpp"
#include "search/AuxData.hpp"
#include "search/Constants.hpp"
#include "search/SearchParams.hpp"
#include "search/SearchResponse.hpp"
#include "util/Math.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <memory>

namespace beta0 {

/*
 * The beta0::Player uses BetaZero MCTS to select actions.
 *
 * Note that when 2 or more identically-configured beta0::Player's are playing in the same
 * game, they can share the same MCTS tree, as an optimization. This implementation supports this
 * optimization.
 */
template <beta0::concepts::Spec Spec_>
class Player : public core::AbstractPlayer<typename Spec_::Game> {
 public:
  using BasePlayer = Player;  // needed for beta0::PlayerGeneratorBase
  using Spec = Spec_;
  using Game = Spec::Game;

  struct Params {
    Params(search::Mode);

    auto make_options_description();

    int num_fast_iters;
    int num_full_iters;
    float full_pct;
    float starting_move_temperature;
    float ending_move_temperature = 0.2;
    float move_temperature_half_life = 0.5 * Spec::MctsConfiguration::kOpeningLength;
    int verbose_num_rows_to_display = core::kNumRowsToDisplayVerbose;
  };

  using Manager = beta0::Manager<Spec>;
  using SearchResults = beta0::SearchResults<Spec>;
  using SearchResponse = search::SearchResponse<SearchResults>;

  using State = Game::State;
  using InfoSet = Game::InfoSet;
  using Move = Game::Move;
  using MoveSet = Game::MoveSet;
  using ActionPrinter = core::ActionPrinter<Game>;
  using ActionRequest = core::ActionRequest<Game>;
  using ActionResponse = core::ActionResponse<Game>;
  using PolicyEncoding = Spec::TensorEncodings::PolicyEncoding;
  using InputFrame = Spec::InputFrame;
  using PolicyTensor = PolicyEncoding::Tensor;
  using GameOutcome = Game::Types::GameOutcome;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using StateChangeUpdate = core::StateChangeUpdate<Game>;
  using InfoSetIterator = core::InfoSetIterator<Game>;
  using AuxData = search::AuxData<Game>;

  struct SharedData {
    template <typename... Ts>
    SharedData(Ts&&... args) : manager(std::forward<Ts>(args)...) {}

    Manager manager;
    int num_raw_policy_starting_moves = 0;
  };
  using SharedData_sptr = std::shared_ptr<SharedData>;

  Player(const Params&, SharedData_sptr, bool owns_shared_data);

  Manager* get_manager() const { return &shared_data_->manager; }
  bool start_game() override;
  void receive_state_change(const StateChangeUpdate&) override;
  ActionResponse get_action_response(const ActionRequest&) override;
  void end_game(const InfoSet& state, const GameOutcome& results) override;

 protected:
  void clear_search_mode();
  void init_search_mode(const ActionRequest&);

  virtual ActionResponse get_action_response_helper(const SearchResults*, const ActionRequest&);

  virtual PolicyTensor get_action_policy(const SearchResults*, const MoveSet&) const;

  void raw_init(const SearchResults*, const MoveSet&, PolicyTensor& policy) const;
  void apply_temperature(PolicyTensor& policy) const;
  void normalize(const InputFrame&, const MoveSet&, PolicyTensor& policy) const;

  core::SearchMode get_random_search_mode() const;

  const Params params_;

  const search::SearchParams search_params_[core::kNumSearchModes];
  math::ExponentialDecay move_temperature_;
  SharedData_sptr shared_data_;
  const bool owns_shared_data_;

  mutable mit::mutex search_mode_mutex_;
  core::SearchMode search_mode_ = core::kNumSearchModes;
  std::vector<AuxData*> aux_data_ptrs_;
};

}  // namespace beta0

#include "inline/beta0/Player.inl"
