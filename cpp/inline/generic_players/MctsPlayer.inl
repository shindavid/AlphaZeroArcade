#include <generic_players/MctsPlayer.hpp>

#include <unistd.h>

#include <core/GameVars.hpp>
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
    num_fast_iters = 1600;
    num_full_iters = 0;
    full_pct = 0.0;
    move_temperature_str = "0.5->0.2:2*sqrt(b)";
  } else if (mode == mcts::kTraining) {
    num_fast_iters = 100;
    num_full_iters = 600;
    full_pct = 0.25;
    move_temperature_str = "0.8->0.2:2*sqrt(b)";
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
    util::KeyValueDumper::add("MctsPlayer num fast iters", "%.8g", full_pct);
    util::KeyValueDumper::add("MctsPlayer move temperature", "%s", move_temperature_str.c_str());
  }
}

template <core::concepts::Game Game>
auto MctsPlayer<Game>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("MctsPlayer options");

  return desc
      .template add_option<"num-fast-iters", 'i'>(
          po::value<int>(&num_fast_iters)->default_value(num_fast_iters),
          "num mcts iterations to do per fast move")
      .template add_option<"num-full-iters", 'I'>(
          po::value<int>(&num_full_iters)->default_value(num_full_iters),
          "num mcts iterations to do per full move")
      .template add_option<"full-pct", 'f'>(po2::float_value("%.2f", &full_pct, full_pct),
                                            "pct of moves that should be full")
      .template add_option<"mean-raw-moves", 'r'>(
          po2::float_value("%.2f", &mean_raw_moves, mean_raw_moves),
          "mean number of raw policy moves to make at the start of each game")
      .template add_option<"move-temp", 't'>(
          po::value<std::string>(&move_temperature_str)->default_value(move_temperature_str),
          "temperature for move selection")
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
MctsPlayer<Game>::MctsPlayer(const Params& params, const mcts::ManagerParams& manager_params)
    : MctsPlayer(params) {
  mcts_manager_ = new MctsManager(manager_params);
  shared_data_ = new SharedData();
  owns_manager_ = true;
  mcts_manager_->set_player_data(shared_data_);
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
          {params.num_fast_iters, true},  // kFast
          {params.num_full_iters},        // kFull
          {1, true}                       // kRawPolicy
      },
      move_temperature_(math::ExponentialDecay::parse(params.move_temperature_str,
                                                      core::GameVars<Game>::get_bindings())) {
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
inline void MctsPlayer<Game>::receive_state_change(core::seat_index_t seat, const FullState& state,
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
      IO::print_state(state, action, &this->get_player_names());
    }
  }
}

template <core::concepts::Game Game>
core::ActionResponse MctsPlayer<Game>::get_action_response(const FullState& state,
                                                           const ActionMask& valid_actions) {
  core::SearchMode search_mode = choose_search_mode();
  const SearchResults* mcts_results = mcts_search(state, search_mode);
  return get_action_response_helper(search_mode, mcts_results, valid_actions);
}

template <core::concepts::Game Game>
inline const typename MctsPlayer<Game>::SearchResults* MctsPlayer<Game>::mcts_search(
    const FullState& state, core::SearchMode search_mode) const {
  return mcts_manager_->search(state, search_params_[search_mode]);
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
  PolicyTensor policy;
  auto& policy_array = eigen_util::reinterpret_as_array(policy);
  if (search_mode == core::kRawPolicy) {
    ActionMask valid_actions_subset = valid_actions;
    bitset_util::randomly_zero_out(valid_actions_subset, valid_actions_subset.count() / 2);

    policy_array.setConstant(0);

    for (int a : bitset_util::on_indices(valid_actions_subset)) {
      policy_array(a) = mcts_results->policy_prior(a);
    }
  } else {
    policy = mcts_results->counts;
  }

  if (search_mode != core::kRawPolicy) {
    float temp = move_temperature_.value();
    if (temp != 0) {
      eigen_util::normalize(policy);  // normalize to avoid numerical issues with annealing.
      policy = policy.pow(1.0 / temp);
    } else {
      /*
       * This is awkward, but I couldn't get a simpler incantation to work. I want to do:
       *
       * policy = (policy == policy.maximum()).template cast<torch_util::dtype>();
       *
       * But the above doesn't work.
       */
      PolicyTensor policy_max = policy.maximum();
      PolicyTensor policy_max_broadcasted;
      policy_max_broadcasted.setConstant(policy_max(0));
      policy = (policy == policy_max_broadcasted).template cast<torch_util::dtype>();
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
  core::action_t action = eigen_util::sample(policy)[0];
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
  IO::print_mcts_results(action_policy, mcts_results);
  std::cout << std::endl;
}

}  // namespace generic
