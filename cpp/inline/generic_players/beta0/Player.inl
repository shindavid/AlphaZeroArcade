#include "generic_players/beta0/Player.hpp"

#include "core/Constants.hpp"
#include "util/Asserts.hpp"

#include <unistd.h>

namespace generic::beta0 {

template <search::concepts::Traits Traits>
Player<Traits>::Params::Params(search::Mode mode) {
  if (mode == search::kCompetition) {
    num_fast_iters = 0;
    num_full_iters = 1600;
    full_pct = 1.0;
    starting_move_temperature = 0.5;
  } else if (mode == search::kTraining) {
    num_fast_iters = 100;
    num_full_iters = 600;
    full_pct = 0.25;
    starting_move_temperature = 0.8;
  } else {
    throw util::Exception("Unknown search::Mode: {}", (int)mode);
  }
}

template <search::concepts::Traits Traits>
void Player<Traits>::Params::dump() const {
  if (full_pct == 0) {
    util::KeyValueDumper::add("generic::beta0::Player num iters", "%d", num_fast_iters);
  } else {
    util::KeyValueDumper::add("generic::beta0::Player num fast iters", "%d", num_fast_iters);
    util::KeyValueDumper::add("generic::beta0::Player num full iters", "%d", num_full_iters);
    util::KeyValueDumper::add("generic::beta0::Player pct full iters", "%6.2%%", 100. * full_pct);
  }
}

template <search::concepts::Traits Traits>
auto Player<Traits>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("Player options");

  return desc
    .template add_option<"num-fast-iters">(
      po::value<int>(&num_fast_iters)->default_value(num_fast_iters),
      "num MCTS iterations to do per fast move")
    .template add_option<"num-full-iters", 'i'>(
      po::value<int>(&num_full_iters)->default_value(num_full_iters),
      "num MCTS iterations to do per full move")
    .template add_option<"full-pct", 'f'>(po2::default_value("{:.2f}", &full_pct, full_pct),
                                          "pct of moves that should be full")
    .template add_hidden_option<"starting-move-temp">(
      po::value<float>(&starting_move_temperature)->default_value(starting_move_temperature),
      "starting temperature for move selection")
    .template add_hidden_option<"ending-move-temp">(
      po::value<float>(&ending_move_temperature)->default_value(ending_move_temperature),
      "ending temperature for move selection")
    .template add_option<"move-temp-half-life", 't'>(
      po::value<float>(&move_temperature_half_life)->default_value(move_temperature_half_life),
      "half-life for move temperature")
    .template add_option<"lcb-z-score">(po::value<float>(&LCB_z_score)->default_value(LCB_z_score),
                                        "z-score for LCB. If zero, disable LCB")
    .template add_option<"verbose", 'v'>(po::bool_switch(&verbose)->default_value(verbose),
                                         "MCTS player verbose mode")
    .template add_option<"verbose-num-rows-to-display", 'r'>(
      po::value<int>(&verbose_num_rows_to_display)->default_value(verbose_num_rows_to_display),
      "MCTS player number of rows to display in verbose mode");
}

template <search::concepts::Traits Traits>
inline Player<Traits>::Player(const Params& params, SharedData_sptr shared_data,
                              bool owns_shared_data)
    : params_(params),
      search_params_{
        {params.num_fast_iters, false},  // kFast
        {params.num_full_iters},         // kFull
        {1, false}                       // kRawPolicy
      },
      move_temperature_(params.starting_move_temperature, params.ending_move_temperature,
                        params.move_temperature_half_life),
      shared_data_(shared_data),
      owns_shared_data_(owns_shared_data) {
  if (params.verbose) {
    verbose_info_ = new VerboseInfo();
  }

  RELEASE_ASSERT(shared_data_.get() != nullptr);
}

template <search::concepts::Traits Traits>
inline Player<Traits>::~Player() {
  if (verbose_info_) {
    delete verbose_info_;
  }
}

template <search::concepts::Traits Traits>
inline bool Player<Traits>::start_game() {
  clear_search_mode();
  move_temperature_.reset();
  if (owns_shared_data_) {
    get_manager()->start();
  }
  return true;
}

template <search::concepts::Traits Traits>
inline void Player<Traits>::receive_state_change(core::seat_index_t seat, const State& state,
                                                 core::action_t action) {
  clear_search_mode();
  move_temperature_.step();
  if (owns_shared_data_) {
    get_manager()->receive_state_change(seat, state, action);
  }
  if (base_t::get_my_seat() == seat && params_.verbose) {
    if (VerboseManager::get_instance()->auto_terminal_printing_enabled()) {
      IO::print_state(std::cout, state, action, &this->get_player_names());
    }
  }
}

template <search::concepts::Traits Traits>
typename Player<Traits>::ActionResponse Player<Traits>::get_action_response(
  const ActionRequest& request) {
  init_search_mode(request);
  search::SearchRequest search_request(request.notification_unit);
  SearchResponse response = get_manager()->search(search_request);

  if (response.yield_instruction == core::kYield) {
    return ActionResponse::yield(response.extra_enqueue_count);
  } else if (response.yield_instruction == core::kDrop) {
    return ActionResponse::drop();
  }

  return get_action_response_helper(response.results, request);
}

template <search::concepts::Traits Traits>
void Player<Traits>::clear_search_mode() {
  mit::unique_lock lock(search_mode_mutex_);
  search_mode_ = core::kNumSearchModes;
}

template <search::concepts::Traits Traits>
void Player<Traits>::init_search_mode(const ActionRequest& request) {
  mit::unique_lock lock(search_mode_mutex_);
  if (search_mode_ != core::kNumSearchModes) return;

  search_mode_ = request.play_noisily ? core::kRawPolicy : get_random_search_mode();
  get_manager()->set_search_params(search_params_[search_mode_]);
}

template <search::concepts::Traits Traits>
typename Player<Traits>::ActionResponse Player<Traits>::get_action_response_helper(
  const SearchResults* mcts_results, const ActionRequest& request) {
  PolicyTensor modified_policy = this->get_action_policy(mcts_results, request.valid_actions);

  if (this->verbose_info_) {
    this->verbose_info_->action_policy = modified_policy;
    this->verbose_info_->mcts_results = *mcts_results;
    this->verbose_info_->initialized = true;
  }
  core::action_t action = eigen_util::sample(modified_policy);
  RELEASE_ASSERT(request.valid_actions[action]);
  return action;
}

template <search::concepts::Traits Traits>
auto Player<Traits>::get_action_policy(const SearchResults* mcts_results,
                                       const ActionMask& valid_actions) const {
  PolicyTensor policy;
  if (this->search_mode_ == core::kRawPolicy) {
    this->raw_init(mcts_results, valid_actions, policy);
  } else {
    policy = mcts_results->policy_posterior;
    policy = mcts_results->action_symmetry_table.collapse(policy);
    this->apply_temperature(policy);
    policy = mcts_results->action_symmetry_table.symmetrize(policy);
    if (this->params_.LCB_z_score) {
      this->apply_LCB(mcts_results, valid_actions, policy);
    }
  }

  this->normalize(valid_actions, policy);
  return policy;
}

}  // namespace generic::beta0
