#include <generic_players/MctsPlayer.hpp>

#include <unistd.h>

#include <util/Asserts.hpp>
#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/Exception.hpp>
#include <util/Math.hpp>
#include <util/KeyValueDumper.hpp>
#include <util/Random.hpp>
#include <util/RepoUtil.hpp>
#include <util/ScreenUtil.hpp>
#include <util/StringUtil.hpp>
#include <util/TorchUtil.hpp>

namespace generic {

template <core::concepts::Game Game>
MctsPlayer<Game>::Params::Params(mcts::Mode mode) {
  if (mode == mcts::kCompetitive) {
    num_fast_iters = 0;
    num_full_iters = 1600;
    full_pct = 1.0;
    starting_move_temperature = 0.5;
  } else if (mode == mcts::kTraining) {
    num_fast_iters = 100;
    num_full_iters = 600;
    full_pct = 0.25;
    starting_move_temperature = 0.8;
  } else {
    throw util::Exception("Unknown mcts::Mode: %d", (int)mode);
  }
}

template <core::concepts::Game Game>
void MctsPlayer<Game>::Params::dump() const {
  if (full_pct == 0) {
    util::KeyValueDumper::add("MctsPlayer num iters", "%d", num_fast_iters);
  } else {
    util::KeyValueDumper::add("MctsPlayer num fast iters", "%d", num_fast_iters);
    util::KeyValueDumper::add("MctsPlayer num full iters", "%d", num_full_iters);
    util::KeyValueDumper::add("MctsPlayer pct full iters", "%6.2%%", 100. * full_pct);
  }
}

template <core::concepts::Game Game>
auto MctsPlayer<Game>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("MctsPlayer options");

  return desc
      .template add_option<"num-fast-iters">(
          po::value<int>(&num_fast_iters)->default_value(num_fast_iters),
          "num mcts iterations to do per fast move")
      .template add_option<"num-full-iters", 'i'>(
          po::value<int>(&num_full_iters)->default_value(num_full_iters),
          "num mcts iterations to do per full move")
      .template add_option<"full-pct", 'f'>(po2::float_value("%.2f", &full_pct, full_pct),
                                            "pct of moves that should be full")
      .template add_option<"mean-raw-moves", 'r'>(
          po2::float_value("%.2f", &mean_raw_moves, mean_raw_moves),
          "mean number of raw policy moves to make at the start of each game")
      .template add_hidden_option<"starting-move-temp">(
          po::value<float>(&starting_move_temperature)->default_value(starting_move_temperature),
          "starting temperature for move selection")
      .template add_hidden_option<"ending-move-temp">(
          po::value<float>(&ending_move_temperature)->default_value(ending_move_temperature),
          "ending temperature for move selection")
      .template add_option<"move-temp-half-life", 't'>(
          po::value<float>(&move_temperature_half_life)->default_value(move_temperature_half_life),
          "half-life for move temperature")
      .template add_option<"lcb-z-score">(
          po::value<float>(&LCB_z_score)->default_value(LCB_z_score),
          "z-score for LCB. If zero, disable LCB")
      .template add_option<"verbose", 'v'>(po::bool_switch(&verbose)->default_value(verbose),
                                           "mcts player verbose mode");
}

template <core::concepts::Game Game>
inline MctsPlayer<Game>::MctsPlayer(const Params& params, SharedData_sptr shared_data,
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
  if (owns_shared_data_) {
    shared_data->manager.start_threads();
  }

  if (params.verbose) {
    verbose_info_ = new VerboseInfo();
  }

  util::release_assert(shared_data_.get() != nullptr);
}

template <core::concepts::Game Game>
inline MctsPlayer<Game>::~MctsPlayer() {
  if (verbose_info_) {
    delete verbose_info_;
  }
}

template <core::concepts::Game Game>
inline void MctsPlayer<Game>::start_game() {
  move_count_ = 0;
  move_temperature_.reset();
  if (owns_shared_data_) {
    get_manager()->start();
    if (params_.mean_raw_moves) {
      shared_data_->num_raw_policy_starting_moves =
          util::Random::exponential(1.0 / params_.mean_raw_moves);
    }
  }
}

template <core::concepts::Game Game>
inline void MctsPlayer<Game>::receive_state_change(core::seat_index_t seat, const State& state,
                                                    core::action_t action) {
  move_count_++;
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

template <core::concepts::Game Game>
typename MctsPlayer<Game>::ActionResponse MctsPlayer<Game>::get_action_response(const State& state,
                                                           const ActionMask& valid_actions) {
  core::SearchMode search_mode = choose_search_mode();
  const SearchResults* mcts_results = mcts_search(search_mode);
  return get_action_response_helper(search_mode, mcts_results, valid_actions);
}

template <core::concepts::Game Game>
inline const typename MctsPlayer<Game>::SearchResults* MctsPlayer<Game>::mcts_search(
    core::SearchMode search_mode) const {
  return get_manager()->search(search_params_[search_mode]);
}

template <core::concepts::Game Game>
inline core::SearchMode MctsPlayer<Game>::choose_search_mode() const {
  bool use_raw_policy = move_count_ < shared_data_->num_raw_policy_starting_moves;
  return use_raw_policy ? core::kRawPolicy : get_random_search_mode();
}

template <core::concepts::Game Game>
typename MctsPlayer<Game>::ActionResponse MctsPlayer<Game>::get_action_response_helper(
    core::SearchMode search_mode, const SearchResults* mcts_results,
    const ActionMask& valid_actions) const {

  PolicyTensor modified_policy = get_action_policy(search_mode, mcts_results, valid_actions);

  if (verbose_info_) {
    verbose_info_->action_policy = modified_policy;
    verbose_info_->mcts_results = *mcts_results;
    verbose_info_->initialized = true;
  }
  core::action_t action = eigen_util::sample(modified_policy);

  ActionTypeDispatcher::call(valid_actions.index(), [&](auto action_type) {
    constexpr int A = decltype(action_type)::value;
    util::release_assert(std::get<A>(valid_actions)[action], "Invalid action: %d", action);
  });

  return action;
}

template <core::concepts::Game Game>
auto MctsPlayer<Game>::get_action_policy(core::SearchMode search_mode,
                                         const SearchResults* mcts_results,
                                         const ActionMask& valid_actions) const {
  return ActionTypeDispatcher::call(valid_actions.index(), [&](auto action_type) {
    constexpr int A = decltype(action_type)::value;
    const auto& bitset = std::get<A>(valid_actions);
    return get_action_policy_helper(search_mode, mcts_results, bitset);
  });
}

template <core::concepts::Game Game>
template <typename Bitset>
auto MctsPlayer<Game>::get_action_policy_helper(core::SearchMode search_mode,
                                                const SearchResults* mcts_results,
                                                const Bitset& valid_bitset) const {
  PolicyTensor policy, Q_sum, Q_sq_sum;
  const auto& counts = mcts_results->counts;
  if (search_mode == core::kRawPolicy) {
    policy.setConstant(0);

    Bitset valid_bitset_subset = valid_bitset;
    bitset_util::randomly_zero_out(valid_bitset_subset, valid_bitset_subset.count() / 2);

    for (int a : bitset_util::on_indices(valid_bitset_subset)) {
      policy(a) = mcts_results->policy_prior(a);
    }
  } else {
    policy = counts;
  }

  if (search_mode != core::kRawPolicy) {
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
      for (int a : bitset_util::on_indices(valid_bitset)) {
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

        if (mcts::kEnableSearchDebug) {
          int visited_actions = 0;
          for (int a : bitset_util::on_indices(valid_bitset)) {
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
          for (int a : bitset_util::on_indices(valid_bitset)) {
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

          std::vector<std::string> columns = {"action",  "N",   "P",   "Q",
                                              "Q_sigma", "LCB", "UCB", "P*"};
          auto data = eigen_util::sort_rows(
              eigen_util::concatenate_columns(actions_arr, counts_arr, policy_arr, Q_arr,
                                              Q_sigma_arr, LCB_arr, UCB_arr, policy_masked_arr));

          eigen_util::PrintArrayFormatMap fmt_map;
          fmt_map["action"] = [](float x) { return Game::IO::action_to_str(x); };

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
    for (int a : bitset_util::on_indices(valid_bitset)) {
      policy(a) = 1;
    }
    eigen_util::normalize(policy);
  }
  return policy;
}

template <core::concepts::Game Game>
core::SearchMode MctsPlayer<Game>::get_random_search_mode() const {
  float r = util::Random::uniform_real<float>(0.0f, 1.0f);
  return r < params_.full_pct ? core::kFull : core::kFast;
}

template <core::concepts::Game Game>
inline void MctsPlayer<Game>::verbose_dump() const {
  if (!verbose_info_->initialized) return;

  const auto& action_policy = verbose_info_->action_policy;
  const auto& mcts_results = verbose_info_->mcts_results;

  std::cout << std::endl << "CPU pos eval:" << std::endl;
  IO::print_mcts_results(std::cout, action_policy, mcts_results);
}

}  // namespace generic
