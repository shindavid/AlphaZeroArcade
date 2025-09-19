#include "generic_players/beta0/Player.hpp"

#include "core/Constants.hpp"
#include "search/SearchRequest.hpp"
#include "util/Asserts.hpp"
#include "util/BitSet.hpp"
#include "util/BoostUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/KeyValueDumper.hpp"
#include "util/Math.hpp"
#include "util/Random.hpp"
#include "util/ScreenUtil.hpp"

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
    if (facing_human_tui_player_) {
      util::ScreenClearer::clear_once();
    }
    verbose_dump();
    if (!facing_human_tui_player_) {
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
  PolicyTensor modified_policy = get_action_policy(mcts_results, request.valid_actions);

  if (verbose_info_) {
    verbose_info_->action_policy = modified_policy;
    verbose_info_->mcts_results = *mcts_results;
    verbose_info_->initialized = true;
  }
  core::action_t action = eigen_util::sample(modified_policy);
  RELEASE_ASSERT(request.valid_actions[action]);
  return action;
}

template <search::concepts::Traits Traits>
auto Player<Traits>::get_action_policy(const SearchResults* mcts_results,
                                       const ActionMask& valid_actions) const {
  PolicyTensor policy, Q_sum, Q_sq_sum;
  const auto& counts = mcts_results->counts;
  if (search_mode_ == core::kRawPolicy) {
    ActionMask valid_actions_subset = valid_actions;
    bitset_util::randomly_zero_out(valid_actions_subset, valid_actions_subset.count() / 2);

    policy.setConstant(0);

    for (int a : bitset_util::on_indices(valid_actions_subset)) {
      policy(a) = mcts_results->policy_prior(a);
    }
  } else if (search_params_[search_mode_].tree_size_limit <= 1) {
    policy = mcts_results->policy_prior;
  } else {
    policy = counts;
  }

  if (search_mode_ != core::kRawPolicy && search_params_[search_mode_].tree_size_limit > 1) {
    if (params_.LCB_z_score) {
      Q_sum = mcts_results->Q * policy;
      Q_sq_sum = mcts_results->Q_sq * policy;
      Q_sum = mcts_results->action_symmetry_table.collapse(Q_sum);
      Q_sq_sum = mcts_results->action_symmetry_table.collapse(Q_sq_sum);
    }
    policy = mcts_results->action_symmetry_table.collapse(policy);
    float temp = move_temperature_.value();
    if (temp != 0) {
      eigen_util::normalize(policy);  // normalize to avoid numerical issues with annealing.
      policy = policy.pow(1.0 / temp);
    } else {
      /*
       * This is awkward, but I couldn't get a simpler incantation to work. I want to do:
       *
       * policy = (policy == policy.maximum()).template cast<float>();
       *
       * But the above doesn't work.
       */
      PolicyTensor policy_max_tensor = policy.maximum();
      float policy_max = policy_max_tensor(0);
      if (policy_max > 0) {
        PolicyTensor policy_max_broadcasted;
        policy_max_broadcasted.setConstant(policy_max);
        policy = (policy == policy_max_broadcasted).template cast<float>();
      }
    }
    policy = mcts_results->action_symmetry_table.symmetrize(policy);
    if (params_.LCB_z_score) {
      Q_sum = mcts_results->action_symmetry_table.symmetrize(Q_sum);
      Q_sq_sum = mcts_results->action_symmetry_table.symmetrize(Q_sq_sum);

      PolicyTensor Q = Q_sum / counts;
      PolicyTensor Q_sq = Q_sq_sum / counts;
      PolicyTensor Q_sigma_sq = (Q_sq - Q * Q) / counts;
      Q_sigma_sq = eigen_util::cwiseMax(Q_sigma_sq, 0);  // clip negative values to 0
      PolicyTensor Q_sigma = Q_sigma_sq.sqrt();

      PolicyTensor LCB = Q - params_.LCB_z_score * Q_sigma;

      float policy_max = -1;
      float min_LCB = 0;
      bool min_LCB_set = false;

      // Let S be the set of indices at which policy is maximal. The below loop sets min_LCB to
      // min_{i in S} {LCB(i)}
      for (int a : bitset_util::on_indices(valid_actions)) {
        float p = policy(a);
        if (p <= 0) continue;

        if (p > policy_max) {
          policy_max = p;
          min_LCB = LCB(a);
          min_LCB_set = true;
        } else if (p == policy_max) {
          min_LCB = std::min(min_LCB, LCB(a));
          min_LCB_set = true;
        }
      }

      if (min_LCB_set) {
        PolicyTensor UCB = Q + params_.LCB_z_score * Q_sigma;

        // zero out policy wherever UCB < min_LCB
        auto mask = (UCB >= min_LCB).template cast<float>();
        PolicyTensor policy_masked = policy * mask;

        if (search::kEnableSearchDebug) {
          int visited_actions = 0;
          for (int a : bitset_util::on_indices(valid_actions)) {
            if (counts(a)) visited_actions++;
          }

          LocalPolicyArray actions_arr(visited_actions);
          LocalPolicyArray counts_arr(visited_actions);
          LocalPolicyArray policy_arr(visited_actions);
          LocalPolicyArray Q_arr(visited_actions);
          LocalPolicyArray Q_sigma_arr(visited_actions);
          LocalPolicyArray LCB_arr(visited_actions);
          LocalPolicyArray UCB_arr(visited_actions);
          LocalPolicyArray policy_masked_arr(visited_actions);

          int r = 0;
          for (int a : bitset_util::on_indices(valid_actions)) {
            if (counts(a) == 0) continue;

            actions_arr(r) = a;
            counts_arr(r) = counts(a);
            policy_arr(r) = policy(a);
            Q_arr(r) = Q(a);
            Q_sigma_arr(r) = Q_sigma(a);
            LCB_arr(r) = LCB(a);
            UCB_arr(r) = UCB(a);
            policy_masked_arr(r) = policy_masked(a);

            r++;
          }

          policy_arr /= policy_arr.sum();
          policy_masked_arr /= policy_masked_arr.sum();

          static std::vector<std::string> columns = {"action",  "N",   "P",   "Q",
                                                     "Q_sigma", "LCB", "UCB", "P*"};
          auto data = eigen_util::sort_rows(
            eigen_util::concatenate_columns(actions_arr, counts_arr, policy_arr, Q_arr, Q_sigma_arr,
                                            LCB_arr, UCB_arr, policy_masked_arr));

          core::action_mode_t mode = mcts_results->action_mode;
          eigen_util::PrintArrayFormatMap fmt_map{
            {"action", [&](float x) { return Game::IO::action_to_str(x, mode); }},
          };

          std::cout << std::endl << "Applying LCB:" << std::endl;
          eigen_util::print_array(std::cout, data, columns, &fmt_map);
        }

        policy = policy_masked;
      }
    }
  }

  if (!eigen_util::normalize(policy)) {
    // This can happen if MCTS proves that the position is losing. In this case we just choose a
    // random valid action.
    policy.setConstant(0);
    for (int a : bitset_util::on_indices(valid_actions)) {
      policy(a) = 1;
    }
    eigen_util::normalize(policy);
  }
  return policy;
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
inline void Player<Traits>::verbose_dump() const {
  if (!verbose_info_->initialized) return;

  const auto& action_policy = verbose_info_->action_policy;
  const auto& mcts_results = verbose_info_->mcts_results;
  int num_rows_to_display = params_.verbose_num_rows_to_display;
  Algorithms::print_mcts_results(std::cout, action_policy, mcts_results, num_rows_to_display);
}

}  // namespace generic::beta0
