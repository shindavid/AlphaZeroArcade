#include "alpha0/Player.hpp"

#include "core/Constants.hpp"
#include "search/SearchRequest.hpp"
#include "search/VerboseManager.hpp"
#include "util/Asserts.hpp"
#include "util/BoostUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/Math.hpp"
#include "util/Random.hpp"

#include <unistd.h>

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
Player<Spec>::Params::Params(search::Mode mode) {
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

template <alpha0::concepts::Spec Spec>
auto Player<Spec>::Params::make_options_description() {
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
    .template add_option<"verbose", 'v'>(po::bool_switch(&verbose)->default_value(verbose),
                                         "MCTS player verbose mode")
    .template add_option<"lcb-z-score">(po::value<float>(&LCB_z_score)->default_value(LCB_z_score),
                                        "z-score for LCB. If zero, disable LCB")
    .template add_option<"verbose-num-rows-to-display", 'r'>(
      po::value<int>(&verbose_num_rows_to_display)->default_value(verbose_num_rows_to_display),
      "MCTS player number of rows to display in verbose mode");
}

template <alpha0::concepts::Spec Spec>
Player<Spec>::Player(const Params& params, SharedData_sptr shared_data, bool owns_shared_data)
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

template <alpha0::concepts::Spec Spec>
bool Player<Spec>::start_game() {
  clear_search_mode();
  move_temperature_.reset();
  if (owns_shared_data_) {
    get_manager()->start();
  }
  return true;
}

template <alpha0::concepts::Spec Spec>
void Player<Spec>::receive_state_change(const StateChangeUpdate& update) {
  clear_search_mode();
  move_temperature_.jump_to(update.step());
  if (owns_shared_data_) {
    if (update.is_jump()) {
      get_manager()->backtrack(update.state_it(), update.step());
    } else {
      const State& state = update.state_it()->state;
      get_manager()->receive_state_change(update.seat(), state, *update.move());
    }
  }

  if (this->get_my_seat() == update.seat() && verbose()) {
    auto it = update.state_it();

    if (generic::VerboseManager::get_instance()->auto_terminal_printing_enabled()) {
      Game::IO::print_state(std::cout, it->state, update.move(), &this->get_player_names());
    }

    if (it->aux) {
      AuxData* aux_data = reinterpret_cast<AuxData*>(it->aux);
      generic::VerboseManager::get_instance()->set(aux_data->verbose_data);
    }
  }
}

template <alpha0::concepts::Spec Spec>
typename Player<Spec>::ActionResponse Player<Spec>::get_action_response(
  const ActionRequest& request) {
  if (request.aux) {
    AuxData* aux_data = reinterpret_cast<AuxData*>(request.aux);
    return aux_data->action_response;
  }

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

template <alpha0::concepts::Spec Spec>
void Player<Spec>::clear_search_mode() {
  mit::unique_lock lock(search_mode_mutex_);
  search_mode_ = core::kNumSearchModes;
}

template <alpha0::concepts::Spec Spec>
void Player<Spec>::init_search_mode(const ActionRequest& request) {
  mit::unique_lock lock(search_mode_mutex_);
  if (search_mode_ != core::kNumSearchModes) return;

  search_mode_ = request.play_noisily ? core::kRawPolicy : get_random_search_mode();
  get_manager()->set_search_params(search_params_[search_mode_]);
}

template <alpha0::concepts::Spec Spec>
typename Player<Spec>::ActionResponse Player<Spec>::get_action_response_helper(
  const SearchResults* mcts_results, const ActionRequest& request) {
  PolicyTensor modified_policy = get_action_policy(mcts_results, request.valid_moves);
  ActionResponse action_response(
    PolicyEncoding::to_move(request.state, eigen_util::sample(modified_policy)));

  if (verbose() || this->is_facing_backtracking_opponent()) {
    if (this->is_facing_backtracking_opponent() || aux_data_ptrs_.empty()) {
      aux_data_ptrs_.push_back(new AuxData(action_response));
    }
    AuxData* aux_data = aux_data_ptrs_.back();
    if (verbose()) {
      aux_data->verbose_data = std::make_shared<VerboseData>(modified_policy, *mcts_results,
                                                             params_.verbose_num_rows_to_display);
      generic::VerboseManager::get_instance()->set(aux_data->verbose_data);
    }
    action_response.set_aux(aux_data);
  }
  return action_response;
}

template <alpha0::concepts::Spec Spec>
typename Player<Spec>::PolicyTensor Player<Spec>::get_action_policy(
  const SearchResults* mcts_results, const MoveSet& valid_moves) const {
  PolicyTensor policy;
  const auto& frame = mcts_results->frame;
  const auto& counts = mcts_results->counts;
  if (search_mode_ == core::kRawPolicy) {
    raw_init(mcts_results, valid_moves, policy);
  } else if (search_params_[search_mode_].tree_size_limit <= 1) {
    policy = mcts_results->P;
  } else {
    policy = counts;
  }

  if (search_mode_ != core::kRawPolicy && search_params_[search_mode_].tree_size_limit > 1) {
    policy = mcts_results->action_symmetry_table.collapse(frame, policy);
    apply_temperature(policy);
    policy = mcts_results->action_symmetry_table.symmetrize(frame, policy);
    if (params_.LCB_z_score) {
      apply_LCB(mcts_results, valid_moves, policy);
    }
  }

  normalize(frame, valid_moves, policy);
  return policy;
}

template <alpha0::concepts::Spec Spec>
void Player<Spec>::raw_init(const SearchResults* mcts_results, const MoveSet& valid_moves,
                            PolicyTensor& policy) const {
  int n_valid_moves = valid_moves.size();
  Move moves[n_valid_moves];
  int i = 0;
  for (Move move : valid_moves) {
    moves[i++] = move;
  }

  util::Random::shuffle(moves, moves + valid_moves.size());

  policy.setConstant(0);

  int n_moves_to_use = n_valid_moves - (n_valid_moves / 2);
  for (i = 0; i < n_moves_to_use; ++i) {
    auto index = PolicyEncoding::to_index(mcts_results->frame, moves[i]);
    policy.coeffRef(index) = mcts_results->P.coeff(index);
  }
}

template <alpha0::concepts::Spec Spec>
void Player<Spec>::apply_temperature(PolicyTensor& policy) const {
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

template <alpha0::concepts::Spec Spec>
void Player<Spec>::normalize(const InputFrame& frame, const MoveSet& valid_moves,
                             PolicyTensor& policy) const {
  if (!eigen_util::normalize(policy)) {
    // This can happen if MCTS proves that the position is losing. In this case we just choose a
    // random valid action.
    policy.setConstant(0);
    for (Move move : valid_moves) {
      auto index = PolicyEncoding::to_index(frame, move);
      policy.coeffRef(index) = 1;
    }
    eigen_util::normalize(policy);
  }
}

template <alpha0::concepts::Spec Spec>
core::SearchMode Player<Spec>::get_random_search_mode() const {
  if (params_.full_pct >= 1.0) {
    return core::kFull;
  }
  float r = util::Random::uniform_real<float>(0.0f, 1.0f);
  return r < params_.full_pct ? core::kFull : core::kFast;
}

template <alpha0::concepts::Spec Spec>
void Player<Spec>::end_game(const State& state, const GameOutcome& results) {
  for (auto ptr : aux_data_ptrs_) {
    delete ptr;
  }
  aux_data_ptrs_.clear();
}

template <alpha0::concepts::Spec Spec>
void Player<Spec>::apply_LCB(const SearchResults* mcts_results, const MoveSet& valid_moves,
                             PolicyTensor& policy) const {
  const auto& table = mcts_results->action_symmetry_table;
  const auto& counts = mcts_results->counts;
  const auto& frame = mcts_results->frame;

  PolicyTensor Q_sum = table.collapse(frame, mcts_results->AQs) * policy;
  PolicyTensor Q_sq_sum = table.collapse(frame, mcts_results->AQs_sq) * policy;

  Q_sum = table.symmetrize(frame, Q_sum);
  Q_sq_sum = table.symmetrize(frame, Q_sq_sum);

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
  for (Move move : valid_moves) {
    auto index = PolicyEncoding::to_index(frame, move);
    float p = policy.coeff(index);
    if (p <= 0) continue;

    if (p > policy_max) {
      policy_max = p;
      min_LCB = LCB.coeff(index);
      min_LCB_set = true;
    } else if (p == policy_max) {
      min_LCB = std::min(min_LCB, LCB.coeff(index));
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
      for (Move move : valid_moves) {
        auto index = PolicyEncoding::to_index(frame, move);
        if (counts.coeff(index)) visited_actions++;
      }

      LocalPolicyArray counts_arr(visited_actions);
      LocalPolicyArray policy_arr(visited_actions);
      LocalPolicyArray Q_arr(visited_actions);
      LocalPolicyArray Q_sigma_arr(visited_actions);
      LocalPolicyArray LCB_arr(visited_actions);
      LocalPolicyArray UCB_arr(visited_actions);
      LocalPolicyArray policy_masked_arr(visited_actions);

      int r = 0;
      for (Move move : valid_moves) {
        auto index = PolicyEncoding::to_index(frame, move);
        if (counts.coeff(index) == 0) continue;

        counts_arr(r) = counts.coeff(index);
        policy_arr(r) = policy.coeff(index);
        Q_arr(r) = Q.coeff(index);
        Q_sigma_arr(r) = Q_sigma.coeff(index);
        LCB_arr(r) = LCB.coeff(index);
        UCB_arr(r) = UCB.coeff(index);
        policy_masked_arr(r) = policy_masked.coeff(index);

        r++;
      }

      ActionPrinter printer(valid_moves);
      LocalPolicyArray actions_arr = printer.flat_array();

      policy_arr /= policy_arr.sum();
      policy_masked_arr /= policy_masked_arr.sum();

      static std::vector<std::string> columns = {"action",  "N",   "P",   "Q",
                                                 "Q_sigma", "LCB", "UCB", "P*"};
      auto data = eigen_util::sort_rows(
        eigen_util::concatenate_columns(actions_arr, counts_arr, policy_arr, Q_arr, Q_sigma_arr,
                                        LCB_arr, UCB_arr, policy_masked_arr));

      eigen_util::PrintArrayFormatMap fmt_map;
      printer.update_format_map(fmt_map);

      std::cout << std::endl << "Applying LCB:" << std::endl;
      eigen_util::print_array(std::cout, data, columns, &fmt_map);
    }

    policy = policy_masked;
  }
}

}  // namespace alpha0
