#include "generic_players/x0/Player.hpp"

#include "core/Constants.hpp"
#include "search/SearchRequest.hpp"
#include "util/Asserts.hpp"
#include "util/BoostUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/Math.hpp"
#include "util/Random.hpp"

#include <unistd.h>

namespace generic::x0 {

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
    .template add_option<"verbose", 'v'>(
      po::bool_switch(&verbose)->default_value(verbose), "MCTS player verbose mode");
}

template <search::concepts::Traits Traits>
Player<Traits>::Player(const Params& params, SharedData_sptr shared_data, bool owns_shared_data)
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
  RELEASE_ASSERT(shared_data_.get() != nullptr);
}

template <search::concepts::Traits Traits>
bool Player<Traits>::start_game() {
  clear_search_mode();
  move_temperature_.reset();
  if (owns_shared_data_) {
    get_manager()->start();
  }
  return true;
}

template <search::concepts::Traits Traits>
void Player<Traits>::receive_state_change(const StateChangeUpdate& update) {
  clear_search_mode();
  move_temperature_.jump_to(update.step());
  if (owns_shared_data_) {
    if (update.is_jump()) {
      get_manager()->backtrack(update.state_it(), update.step());
    } else {
      const State& state = update.state_it()->state;
      get_manager()->receive_state_change(update.seat(), state, update.action());
    }
  }
}

template <search::concepts::Traits Traits>
core::ActionResponse Player<Traits>::get_action_response(const ActionRequest& request) {
  if (request.aux) {
    AuxData* aux_data = reinterpret_cast<AuxData*>(request.aux);
    return aux_data->action_response;
  }

  init_search_mode(request);
  search::SearchRequest search_request(request.notification_unit);
  SearchResponse response = get_manager()->search(search_request);

  if (response.yield_instruction == core::kYield) {
    return core::ActionResponse::yield(response.extra_enqueue_count);
  } else if (response.yield_instruction == core::kDrop) {
    return core::ActionResponse::drop();
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
core::ActionResponse Player<Traits>::get_action_response_helper(const SearchResults* mcts_results,
                                                                const ActionRequest& request) {
  PolicyTensor modified_policy = get_action_policy(mcts_results, request.valid_actions);
  return eigen_util::sample(modified_policy);
}

template <search::concepts::Traits Traits>
void Player<Traits>::raw_init(const SearchResults* mcts_results, const ActionMask& valid_actions,
                              PolicyTensor& policy) const {
  ActionMask valid_actions_subset = valid_actions;
  valid_actions_subset.randomly_zero_out(valid_actions_subset.count() / 2);

  policy.setConstant(0);

  for (int a : valid_actions_subset.on_indices()) {
    policy(a) = mcts_results->P(a);
  }
}

template <search::concepts::Traits Traits>
void Player<Traits>::apply_temperature(PolicyTensor& policy) const {
  float temp = move_temperature_.value();
  if (temp != 0) {
    eigen_util::normalize(policy);  // normalize to avoid numerical issues with annealing.
    policy = policy.pow(1.0 / temp);
  } else {
    float policy_max = eigen_util::max(policy);
    if (policy_max > 0) {
      PolicyTensor policy_max_broadcasted;
      policy_max_broadcasted.setConstant(policy_max);
      policy = (policy == policy_max_broadcasted).template cast<float>();
    }
  }
}

template <search::concepts::Traits Traits>
void Player<Traits>::normalize(const ActionMask& valid_actions, PolicyTensor& policy) const {
  if (!eigen_util::normalize(policy)) {
    // This can happen if MCTS proves that the position is losing. In this case we just choose a
    // random valid action.
    policy.setConstant(0);
    for (int a : valid_actions.on_indices()) {
      policy(a) = 1;
    }
    eigen_util::normalize(policy);
  }
}

template <search::concepts::Traits Traits>
core::SearchMode Player<Traits>::get_random_search_mode() const {
  if (params_.full_pct >= 1.0) {
    return core::kFull;
  }
  float r = util::Random::uniform_real<float>(0.0f, 1.0f);
  return r < params_.full_pct ? core::kFull : core::kFast;
}

template <search::concepts::Traits Traits>
void Player<Traits>::end_game(const State& state, const GameResultTensor& results) {
  for (auto ptr : aux_data_ptrs_) {
    delete ptr;
  }
  aux_data_ptrs_.clear();
}

}  // namespace generic::x0
