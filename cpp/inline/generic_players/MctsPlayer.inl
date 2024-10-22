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
inline MctsPlayer<Game>::MctsPlayer(const Params& params, MctsManager* mcts_manager)
    : MctsPlayer(params) {
  mcts_manager_ = mcts_manager;
  shared_data_ = (SharedData*)mcts_manager->get_player_data();
  owns_manager_ = false;

  util::release_assert(mcts_manager_ != nullptr);
  util::release_assert(shared_data_ != nullptr);
}

template <core::concepts::Game Game>
MctsPlayer<Game>::MctsPlayer(const Params& params, const MctsManagerParams& manager_params)
    : MctsPlayer(params) {
  mcts_manager_ = new MctsManager(manager_params);
  shared_data_ = new SharedData();
  owns_manager_ = true;
  mcts_manager_->set_player_data(shared_data_);
  mcts_manager_->start_threads();
}

template <core::concepts::Game Game>
inline MctsPlayer<Game>::~MctsPlayer() {
  if (verbose_info_) {
    delete verbose_info_;
  }
  if (owns_manager_) {
    delete mcts_manager_;
    delete shared_data_;
  }
}

template <core::concepts::Game Game>
MctsPlayer<Game>::MctsPlayer(const Params& params)
    : params_(params),
      search_params_{
          {params.num_fast_iters, false},  // kFast
          {params.num_full_iters},         // kFull
          {1, false}                       // kRawPolicy
      },
      move_temperature_(params.starting_move_temperature, params.ending_move_temperature,
                        params.move_temperature_half_life) {
  if (params.verbose) {
    verbose_info_ = new VerboseInfo();
  }
}

template <core::concepts::Game Game>
inline void MctsPlayer<Game>::start_game() {
  move_count_ = 0;
  move_temperature_.reset();
  if (owns_manager_) {
    mcts_manager_->start();
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
  if (owns_manager_) {
    mcts_manager_->receive_state_change(seat, state, action);
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
core::ActionResponse MctsPlayer<Game>::get_action_response(const State& state,
                                                           const ActionMask& valid_actions) {
  core::SearchMode search_mode = choose_search_mode();
  const SearchResults* mcts_results = mcts_search(search_mode);
  return get_action_response_helper(search_mode, mcts_results, valid_actions);
}

template <core::concepts::Game Game>
inline const typename MctsPlayer<Game>::SearchResults* MctsPlayer<Game>::mcts_search(
    core::SearchMode search_mode) const {
  return mcts_manager_->search(search_params_[search_mode]);
}

template <core::concepts::Game Game>
inline core::SearchMode MctsPlayer<Game>::choose_search_mode() const {
  bool use_raw_policy = move_count_ < shared_data_->num_raw_policy_starting_moves;
  return use_raw_policy ? core::kRawPolicy : get_random_search_mode();
}

template <core::concepts::Game Game>
core::ActionResponse MctsPlayer<Game>::get_action_response_helper(
    core::SearchMode search_mode, const SearchResults* mcts_results,
    const ActionMask& valid_actions) const {
  PolicyTensor policy, Q_sum, Q_sq_sum;
  const auto& counts = mcts_results->counts;
  auto& policy_array = eigen_util::reinterpret_as_array(policy);
  if (search_mode == core::kRawPolicy) {
    ActionMask valid_actions_subset = valid_actions;
    bitset_util::randomly_zero_out(valid_actions_subset, valid_actions_subset.count() / 2);

    policy_array.setConstant(0);

    for (int a : bitset_util::on_indices(valid_actions_subset)) {
      policy_array(a) = mcts_results->policy_prior(a);
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
      auto LCB_array = eigen_util::reinterpret_as_array(LCB);

      float policy_max = -1;
      float min_LCB = 0;
      bool min_LCB_set = false;

      // Let S be the set of indices at which policy is maximal. The below loop sets min_LCB to
      // min_{i in S} {LCB_array[i])}
      for (int a : bitset_util::on_indices(valid_actions)) {
        float p = policy_array(a);
        if (p <= 0) continue;

        if (p > policy_max) {
          policy_max = p;
          min_LCB = LCB_array(a);
          min_LCB_set = true;
        } else if (p == policy_max) {
          min_LCB = std::min(min_LCB, LCB_array(a));
          min_LCB_set = true;
        }
      }

      if (min_LCB_set) {
        PolicyTensor UCB = Q + params_.LCB_z_score * Q_sigma;
        auto UCB_array = eigen_util::reinterpret_as_array(UCB);

        // zero out policy_array wherever UCB < min_LCB
        auto mask = (UCB_array >= min_LCB).template cast<float>();
        auto policy_masked = policy_array * mask;

        if (mcts::kEnableSearchDebug) {
          const char* header[] = {"action", "N", "P", "Q", "Q_sigma", "LCB", "UCB", "P*"};
          constexpr int nColumns = sizeof(header) / sizeof(header[0]);

          int visited_actions = 0;
          for (int a : bitset_util::on_indices(valid_actions)) {
            if (counts(a)) visited_actions++;
          }

          std::ostringstream ss;
          using Array = Eigen::Array<float, Eigen::Dynamic, nColumns, 0,
                                     Game::Constants::kMaxBranchingFactor>;
          Array arr(visited_actions, nColumns);
          int r = 0;
          for (int a : bitset_util::on_indices(valid_actions)) {
            if (counts(a) == 0) continue;

            int c = 0;
            arr(r, c++) = a;
            arr(r, c++) = counts(a);
            arr(r, c++) = policy(a);
            arr(r, c++) = Q(a);
            arr(r, c++) = Q_sigma(a);
            arr(r, c++) = LCB(a);
            arr(r, c++) = UCB(a);
            arr(r, c++) = policy_masked(a);

            r++;
            util::release_assert(c == nColumns);
          }

          ss << arr;
          std::string s = ss.str();
          std::string first_line = s.substr(0, s.find('\n'));
          int column_width = (first_line.size() - nColumns + 1) / nColumns;
          std::string fmt = util::create_string("%%%ds ", column_width);

          std::ostringstream ss2;
          for (int i = 0; i < nColumns; i++) {
            ss2 << util::create_string(fmt.c_str(), header[i]);
          }
          LOG_INFO << "visited_actions: " << visited_actions;
          LOG_INFO << "Applying LCB:\n" << ss2.str() << "\n" << s;
        }

        policy_array = policy_masked;
      }
    }
  }

  if (!eigen_util::normalize(policy)) {
    // This can happen if MCTS proves that the position is losing. In this case we just choose a
    // random valid action.
    policy_array.setConstant(0);
    for (int a : bitset_util::on_indices(valid_actions)) {
      policy_array(a) = 1;
    }
    eigen_util::normalize(policy);
  }

  if (verbose_info_) {
    verbose_info_->action_policy = policy;
    verbose_info_->mcts_results = *mcts_results;
    verbose_info_->initialized = true;
  }
  core::action_t action = eigen_util::sample(policy_array);
  util::release_assert(valid_actions[action]);
  return action;
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

  printf("CPU pos eval:\n");
  IO::print_mcts_results(std::cout, action_policy, mcts_results);
}

}  // namespace generic
