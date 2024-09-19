#pragma once

#include <ostream>

#include <boost/filesystem.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/Constants.hpp>
#include <core/concepts/Game.hpp>
#include <mcts/Constants.hpp>
#include <mcts/Manager.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/SearchParams.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/Math.hpp>

namespace generic {

/*
 * The MctsPlayer uses MCTS to select actions.
 *
 * Note that when 2 or more identically-configured MctsPlayer's are playing in the same game, they
 * can share the same MCTS tree, as an optimization. This implementation supports this optimization.
 */
template <core::concepts::Game Game>
class MctsPlayer : public core::AbstractPlayer<Game> {
 public:
  using base_t = core::AbstractPlayer<Game>;

  struct Params {
    Params(mcts::Mode);
    void dump() const;

    auto make_options_description();

    int num_fast_iters;
    int num_full_iters;
    float full_pct;
    float mean_raw_moves = 0.0;
    float starting_move_temperature;
    float ending_move_temperature = 0.2;
    float move_temperature_half_life = 0.5 * Game::Constants::kOpeningLength;
    bool verbose = false;
  };

  using MctsManager = mcts::Manager<Game>;
  using MctsManagerParams = mcts::ManagerParams<Game>;
  using MctsSearchParams = mcts::SearchParams;
  using SearchResults = Game::Types::SearchResults;
  using player_name_array_t = Game::Types::player_name_array_t;

  using FullState = Game::FullState;
  using IO = Game::IO;
  using ActionMask = Game::Types::ActionMask;
  using ValueArray = Game::Types::ValueArray;
  using PolicyTensor = Game::Types::PolicyTensor;

  // uses this constructor when sharing an MCTS manager
  MctsPlayer(const Params&, MctsManager* mcts_manager);
  MctsPlayer(const Params&, const MctsManagerParams& manager_params);
  ~MctsPlayer();

  MctsManager* get_mcts_manager() { return mcts_manager_; }
  void start_game() override;
  void receive_state_change(core::seat_index_t, const FullState&, core::action_t) override;
  core::ActionResponse get_action_response(const FullState&, const ActionMask&) override;
  void set_facing_human_tui_player() override {
    facing_human_tui_player_ = true;  // affects printing
  }

 protected:
  MctsPlayer(const Params&);

  const SearchResults* mcts_search(const FullState& state, core::SearchMode search_mode) const;
  core::SearchMode choose_search_mode() const;
  core::ActionResponse get_action_response_helper(core::SearchMode, const SearchResults*,
                                                  const ActionMask& valid_actions) const;

  struct VerboseInfo {
    PolicyTensor action_policy;
    SearchResults mcts_results;

    bool initialized = false;
  };

  struct SharedData {
    int num_raw_policy_starting_moves = 0;
  };

  core::SearchMode get_random_search_mode() const;
  void verbose_dump() const;

  const Params params_;

  const MctsSearchParams search_params_[core::kNumSearchModes];
  math::ExponentialDecay move_temperature_;
  MctsManager* mcts_manager_;
  SharedData* shared_data_ = nullptr;
  VerboseInfo* verbose_info_ = nullptr;
  bool owns_manager_;
  bool facing_human_tui_player_ = false;
  int move_count_ = 0;
};

}  // namespace generic

#include <inline/generic_players/MctsPlayer.inl>
