#include <core/GameServer.hpp>

#include <core/BasicTypes.hpp>
#include <core/Globals.hpp>
#include <core/Packet.hpp>
#include <core/PerfStats.hpp>
#include <core/players/RemotePlayerProxy.hpp>
#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/Exception.hpp>
#include <util/KeyValueDumper.hpp>
#include <util/LoggingUtil.hpp>
#include <util/Random.hpp>
#include <util/SocketUtil.hpp>
#include <util/StringUtil.hpp>

#include <boost/program_options.hpp>

#include <arpa/inet.h>
#include <format>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>

namespace core {

template <concepts::Game Game>
auto GameServer<Game>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("GameServer options");

  return desc
    .template add_option<"port">(po::value<int>(&port)->default_value(port),
                                 "port for external players to connect to (must be set to a "
                                 "nonzero value if using external players)")
    .template add_option<"num-games", 'G'>(po::value<int>(&num_games)->default_value(num_games),
                                           "num games (<=0 means run indefinitely)")
    .template add_option<"parallelism", 'p'>(
      po::value<int>(&parallelism)->default_value(parallelism),
      "max num games to play simultaneously. Participating players can request lower values "
      "and the server will respect their requests")
    .template add_option<"num-game-threads", 't'>(
      po::value<int>(&num_game_threads)->default_value(num_game_threads),
      "num threads to use for running games")
    .template add_option<"mean-noisy-moves", 'n'>(
      po2::default_value("{:.2f}", &mean_noisy_moves, mean_noisy_moves),
      "mean number of noisy moves to make at the start of each game")
    .template add_flag<"display-progress-bar", "hide-progress-bar">(
      &display_progress_bar, "display progress bar (only in tty-mode without TUI player)",
      "hide progress bar")
    .template add_flag<"print-game-states", "do-not-print-game-states">(
      &print_game_states, "print game state between moves", "do not print game state between moves")
    .template add_flag<"announce-game-results", "do-not-announce-game-results">(
      &announce_game_results, "announce result after each individual game",
      "do not announce result after each individual game")
    .template add_hidden_flag<"respect-victory-hints", "do-not-respect-victory-hints">(
      &respect_victory_hints, "immediately exit game if a player claims imminent victory",
      "ignore imminent victory claims from players");
}

template <concepts::Game Game>
GameServer<Game>::SharedData::SharedData(
  GameServer* server, const Params& params,
  const TrainingDataWriterParams& training_data_writer_params)
    : server_(server), params_(params) {
  if (training_data_writer_params.enabled) {
    training_data_writer_ = new TrainingDataWriter(training_data_writer_params);
  }
}

template <concepts::Game Game>
GameServer<Game>::SharedData::~SharedData() {
  yield_manager_.shut_down();

  if (bar_) delete bar_;

  for (auto& reg : registrations_) {
    delete reg.gen;
  }

  for (GameSlot* slot : game_slots_) {
    delete slot;
  }

  delete training_data_writer_;
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::init_slots() {
  int n_slots = params_.parallelism;
  if (params_.num_games > 0) {
    n_slots = std::min(n_slots, params_.num_games);
  }

  for (const auto& reg : registrations_) {
    int n = reg.gen->max_simultaneous_games();
    if (n > 0) n_slots = std::min(n_slots, n);
  }

  for (int p = 0; p < n_slots; ++p) {
    GameSlot* slot = new GameSlot(*this, p);
    game_slots_.push_back(slot);
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::start_games() {
  for (GameSlot* slot : game_slots_) {
    if (!slot->start_game()) {
      throw util::Exception("ERROR: failed to start game slot");
    }
    queue_.emplace(slot->id());
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::init_progress_bar() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (bar_) return;

  if (params_.display_progress_bar && params_.num_games > 0 && util::tty_mode()) {
    bar_ = new progressbar(params_.num_games + 1);  // + 1 for first update
    bar_->update();                                 // so that progress-bar displays immediately
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::init_random_seat_indices() {
  std::bitset<kNumPlayers> fixed_seat_indices;
  for (PlayerRegistration& reg : registrations_) {
    if (reg.seat >= 0) {
      fixed_seat_indices.set(reg.seat);
    }
  }

  for (seat_index_t random_seat : bitset_util::off_indices(fixed_seat_indices)) {
    random_seat_indices_[num_random_seats_++] = random_seat;
  }
  util::Random::shuffle(&random_seat_indices_[0], &random_seat_indices_[num_random_seats_]);
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::run_yield_manager() {
  yield_manager_.run([this](const slot_context_vec_t& slot_contexts) {
    std::unique_lock lock(mutex_);

    for (SlotContext item : slot_contexts) {
      LOG_DEBUG("<-- GameServer::{}(): enqueueing item={}:{}", __func__, item.slot, item.context);
      queue_.push(item);
    }
    pending_queue_count_ -= slot_contexts.size();
    lock.unlock();

    cv_.notify_all();
  });
}

template <concepts::Game Game>
bool GameServer<Game>::SharedData::next(int64_t& wait_for_game_slot_time_ns, SlotContext& item) {
  core::PerfClocker clocker(wait_for_game_slot_time_ns);
  std::unique_lock lock(mutex_);

  if (queue_.empty()) {
    if (pending_queue_count_ > 0) {
      LOG_DEBUG("<-- GameServer::{}(): waiting (queue:{} pending:{})", __func__, queue_.size(),
                pending_queue_count_);
      server_->force_progress();
      cv_.wait(lock, [&] { return paused_ || !queue_pending(); });
    }
    if (queue_.empty()) {
      LOG_DEBUG("<-- GameServer::{}(): queue empty, exiting", __func__);
      return false;
    }
  }

  if (paused_) {
    LOG_DEBUG("<-- GameServer::{}(): paused", __func__);
    return false;
  }

  item = queue_.front();
  queue_.pop();
  pending_queue_count_++;
  LOG_DEBUG("<-- GameServer::{}(): item={}:{} (queue:{} pending:{})", __func__, item.slot,
            item.context, queue_.size(), pending_queue_count_);
  return true;
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::enqueue(SlotContext item, const EnqueueRequest& request) {
  std::unique_lock lock(mutex_);
  bool was_queue_pending = queue_pending();
  if (request.instruction == kEnqueueNow) {
    util::release_assert(request.extra_enqueue_count == 0);
    item.context = 0;  // when continuing, we always want to reset the context to 0
    queue_.push(item);
    pending_queue_count_--;
  } else if (request.instruction == kEnqueueLater) {
    if (request.extra_enqueue_count > 0) {
      // Push back the item's siblings
      for (int i = 0; i < request.extra_enqueue_count; ++i) {
        item.context = i + 1;
        queue_.push(item);
      }
      item.context = 0;  // just for the LOG_DEBUG() statement below
    }
  } else if (request.instruction == kEnqueueNever) {
    pending_queue_count_--;
  } else {
    throw util::Exception("GameServer::{}(): invalid enqueue instruction: {}", __func__,
                          (int)request.instruction);
  }

  LOG_DEBUG("<-- GameServer::{}(item={}:{}, request={}:{}) pending={} queue={}", __func__,
            item.slot, item.context, (int)request.instruction, request.extra_enqueue_count,
            pending_queue_count_, queue_.size());

  lock.unlock();

  if (was_queue_pending && !queue_pending()) {
    cv_.notify_all();
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::drop_slot() {
  std::unique_lock lock(mutex_);
  pending_queue_count_--;
  if (pending_queue_count_ == 0 && queue_.empty()) {
    lock.unlock();
    cv_.notify_all();
  }
}

template <concepts::Game Game>
bool GameServer<Game>::SharedData::request_game() {
  if (LoopControllerClient::deactivated()) return false;
  if (training_data_writer_ && training_data_writer_->closed()) return false;

  std::lock_guard<std::mutex> guard(mutex_);
  if (params_.num_games > 0 && num_games_started_ >= params_.num_games) return false;
  num_games_started_++;
  return true;
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::update(const ValueArray& outcome) {
  std::lock_guard<std::mutex> guard(mutex_);
  for (seat_index_t s = 0; s < kNumPlayers; ++s) {
    results_array_[s][outcome[s]]++;
  }

  if (bar_) bar_->update();
}

template <concepts::Game Game>
auto GameServer<Game>::SharedData::get_results() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return results_array_;
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::start_session() {
  for (auto& reg : registrations_) {
    reg.gen->start_session();
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::end_session() {
  for (auto& reg : registrations_) {
    reg.gen->end_session();
  }
}

template <concepts::Game Game>
bool GameServer<Game>::SharedData::ready_to_start() const {
  for (const auto& reg : registrations_) {
    auto* remote_gen = dynamic_cast<RemotePlayerProxyGenerator*>(reg.gen);
    if (remote_gen && !remote_gen->initialized()) return false;
  }
  return true;
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::register_player(seat_index_t seat, PlayerGenerator* gen,
                                                   bool implicit_remote) {
  util::clean_assert(seat < kNumPlayers, "Invalid seat number {}", seat);
  if (dynamic_cast<RemotePlayerProxyGenerator*>(gen)) {
    if (implicit_remote) {
      util::clean_assert(
        params_.port > 0,
        "If specifying fewer than {} --player's, the remaining players are assumed to be remote "
        "players. In this case, --port must be specified, so that the remote players can connect",
        kNumPlayers);
    } else {
      util::clean_assert(params_.port > 0, "Cannot use remote players without setting --port");
    }

    util::clean_assert(seat < 0, "Cannot specify --seat with --type=Remote");
  }
  if (seat >= 0) {
    for (const auto& reg : registrations_) {
      util::clean_assert(reg.seat != seat, "Double-seated player at seat {}", seat);
    }
  }
  player_id_t player_id = registrations_.size();
  util::clean_assert(player_id < kNumPlayers, "Too many players registered (max {})", kNumPlayers);
  std::string name = gen->get_name();
  if (name.empty()) {
    gen->set_name(gen->get_default_name());
  }
  registrations_.emplace_back(gen, seat, player_id);
}

template <concepts::Game Game>
typename GameServer<Game>::player_instantiation_array_t
GameServer<Game>::SharedData::generate_player_order(
    const player_instantiation_array_t& instantiations) {
  std::unique_lock lock(mutex_);
  std::next_permutation(random_seat_indices_.begin(),
                        random_seat_indices_.begin() + num_random_seats_);
  player_id_array_t random_seat_index_permutation = random_seat_indices_;
  lock.unlock();

  player_instantiation_array_t player_order;

  for (int p = 0; p < kNumPlayers; ++p) {
    int s = registrations_[p].seat;
    if (s < 0) continue;
    player_order[s] = instantiations[p];
  }

  int r = 0;
  for (int p = 0; p < kNumPlayers; ++p) {
    int s = registrations_[p].seat;
    if (s >= 0) continue;
    s = random_seat_index_permutation[r++];
    player_order[s] = instantiations[p];
  }

  for (int p = 0; p < kNumPlayers; ++p) {
    player_order[p].seat = p;
  }

  return player_order;
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::pause() {
  LOG_INFO("GameServer: pausing");
  std::unique_lock lock(mutex_);
  util::release_assert(!paused_);
  paused_ = true;
  pause_receipt_pending_ = true;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::unpause() {
  LOG_INFO("GameServer: unpausing");
  std::unique_lock lock(mutex_);
  util::release_assert(paused_);
  paused_ = false;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::wait_for_unpause() {
  LOG_DEBUG("<-- GameServer: waiting for unpause...");
  std::unique_lock lock(mutex_);
  cv_.wait(lock, [&] { return !paused_; });
  LOG_DEBUG("<-- GameServer: unpause wait complete!");
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::increment_active_thread_count() {
  std::unique_lock lock(mutex_);
  active_thread_count_++;
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::decrement_active_thread_count() {
  std::unique_lock lock(mutex_);
  active_thread_count_--;
  issue_pause_receipt_if_necessary();
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::increment_paused_thread_count() {
  std::unique_lock lock(mutex_);
  util::release_assert(paused_);
  paused_thread_count_++;
  LOG_DEBUG("<-- GameServer: pause_thread_count={} active_thread_count={}", paused_thread_count_,
            active_thread_count_);
  issue_pause_receipt_if_necessary();
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::decrement_paused_thread_count() {
  std::unique_lock lock(mutex_);
  util::release_assert(!paused_);
  if (paused_thread_count_ == 1) {
    core::LoopControllerClient::get()->handle_unpause_receipt();
    paused_thread_count_--;
    lock.unlock();
    cv_.notify_all();
  } else {
    paused_thread_count_--;
    cv_.wait(lock, [&] { return paused_thread_count_ == 0; });
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::issue_pause_receipt_if_necessary() {
  // assumes mutex_ is locked
  if (paused_ && pause_receipt_pending_ && active_thread_count_ == paused_thread_count_) {
    pause_receipt_pending_ = false;
    LOG_DEBUG("<-- GameServer: handling pause receipt");
    core::LoopControllerClient::get()->handle_pause_receipt();
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::update_perf_stats(PerfStats& stats) {
  core::SearchThreadPerfStats search_thread_stats;
  search_thread_stats.wait_for_game_slot_time_ns =
    wait_for_game_slot_time_ns_.exchange(int64_t(0), std::memory_order_relaxed);
  search_thread_stats.mcts_time_ns = mcts_time_ns_.exchange(int64_t(0), std::memory_order_relaxed);
  stats.update(search_thread_stats);
}

template <concepts::Game Game>
GameServer<Game>::GameSlot::GameSlot(SharedData& shared_data, game_slot_index_t id)
    : shared_data_(shared_data),
      id_(id) {
  std::bitset<kNumPlayers> human_tui_indices;
  for (int p = 0; p < kNumPlayers; ++p) {
    instantiations_[p] = shared_data_.registration_templates()[p].instantiate(id);
    human_tui_indices[p] = instantiations_[p].player->is_human_tui_player();
  }

  for (int p = 0; p < kNumPlayers; ++p) {
    std::bitset<kNumPlayers> human_tui_indices_copy(human_tui_indices);
    human_tui_indices_copy[p] = false;
    if (human_tui_indices_copy.any()) {
      instantiations_[p].player->set_facing_human_tui_player();
    }
  }

  if (!human_tui_indices.any()) {
    shared_data_.init_progress_bar();
  }
}

template <concepts::Game Game>
GameServer<Game>::GameSlot::~GameSlot() {
  for (const auto& reg : instantiations_) {
    delete reg.player;
  }
}

template <concepts::Game Game>
GameServerBase::EnqueueRequest GameServer<Game>::GameSlot::step(context_id_t context) {
  EnqueueRequest request;
  if (!Rules::is_chance_mode(action_mode_)) {
    if (step_non_chance(context, request)) {
      pre_step();
    } else {
      return request;
    }
  }

  if (Rules::is_chance_mode(action_mode_)) {
    if (step_chance()) {
      pre_step();
    }
  }
  return request;
}

template <concepts::Game Game>
void GameServer<Game>::GameSlot::pre_step() {
  util::debug_assert(!mid_yield_);
  // Even with multi-threading enabled via ActionResponse::extra_enqueue_count, we should never
  // get here with multiple threads
  action_mode_ = Rules::get_action_mode(state_history_.current());
  noisy_mode_ = move_number_ < num_noisy_starting_moves_;
  if (!Rules::is_chance_mode(action_mode_)) {
    active_seat_ = Rules::get_current_player(state_history_.current());
    valid_actions_ = Rules::get_legal_moves(state_history_);
  }
}

template <concepts::Game Game>
bool GameServer<Game>::GameSlot::step_chance() {
  for (; step_chance_player_index_ < kNumPlayers; ++step_chance_player_index_) {
    Player* player = players_[step_chance_player_index_];
    YieldNotificationUnit notification_unit(shared_data_.yield_manager(), id_, 0);
    ChangeEventPreHandleRequest request(notification_unit);
    ChanceEventPreHandleResponse response = player->prehandle_chance_event(request);
    mid_yield_ = response.yield_instruction != kContinue;
    if (mid_yield_) {
      return false;
    }
    if (!noisy_mode_ && response.action_values) {
      util::release_assert(!chance_action_values_,
                           "Clashing chance-event training info from different players");
    }
    chance_action_values_ = response.action_values;
  }

  ChanceDistribution chance_dist = Rules::get_chance_distribution(state_history_.current());
  action_t action = eigen_util::sample(chance_dist);
  if (game_log_) {
    game_log_->add(state_history_.current(), action, active_seat_, nullptr, chance_action_values_,
                   chance_action_values_);
  }

  // reset for next chance event:
  step_chance_player_index_ = 0;
  chance_action_values_ = nullptr;

  Rules::apply(state_history_, action);
  if (params().print_game_states) {
    Game::IO::print_state(std::cout, state_history_.current(), action, &player_names_);
  }
  for (auto player2 : players_) {
    player2->receive_state_change(active_seat_, state_history_.current(), action);
  }

  ValueTensor outcome;
  if (Game::Rules::is_terminal(state_history_.current(), active_seat_, action, outcome)) {
    handle_terminal(outcome);
    return false;
  }
  return true;
}

template <concepts::Game Game>
bool GameServer<Game>::GameSlot::step_non_chance(context_id_t context,
                                                 EnqueueRequest& enqueue_request) {
  Player* player = players_[active_seat_];
  YieldNotificationUnit notification_unit(shared_data_.yield_manager(), id_, context);
  ActionRequest request(state_history_.current(), valid_actions_, notification_unit);
  request.play_noisily = noisy_mode_;

  ActionResponse response = player->get_action_response(request);
  util::debug_assert(response.extra_enqueue_count == 0 || response.yield_instruction == kYield,
                     "Invalid response: extra={} instr={}", response.extra_enqueue_count,
                     int(response.yield_instruction));

  switch (response.yield_instruction) {
    case kContinue: {
      mid_yield_ = false;
      break;
    }
    case kYield: {
      mid_yield_ = true;
      pending_drop_count_ += response.extra_enqueue_count;
      enqueue_request.instruction = kEnqueueLater;
      enqueue_request.extra_enqueue_count = response.extra_enqueue_count;
      return false;
    }
    case kDrop: {
      pending_drop_count_--;
      enqueue_request.instruction = kEnqueueNever;
      return false;
    }
    default: {
      throw util::Exception("Unexpected response: {}", int(response.yield_instruction));
    }
  }

  util::debug_assert(!mid_yield_);

  move_number_++;
  action_t action = response.action;
  const TrainingInfo& training_info = response.training_info;
  if (game_log_) {
    game_log_->add(state_history_.current(), action, active_seat_, training_info.policy_target,
                   training_info.action_values_target, training_info.use_for_training);
  }

  // TODO: gracefully handle and prompt for retry. Otherwise, a malicious remote process can crash
  // the server.
  util::release_assert(valid_actions_[action], "Invalid action: {}", action);

  if (response.victory_guarantee && params().respect_victory_hints) {
    ValueTensor outcome = GameResults::win(active_seat_);
    if (params().announce_game_results) {
      LOG_INFO("Short-circuiting game {} because player {} (seat={}) claims victory",
              game_id_, player->get_name(), active_seat_);
    }
    handle_terminal(outcome);
    return false;
  } else {
    Rules::apply(state_history_, action);
    if (params().print_game_states) {
      Game::IO::print_state(std::cout, state_history_.current(), action, &player_names_);
    }
    for (auto player2 : players_) {
      player2->receive_state_change(active_seat_, state_history_.current(), action);
    }

    ValueTensor outcome;
    if (Game::Rules::is_terminal(state_history_.current(), active_seat_, action, outcome)) {
      handle_terminal(outcome);
      return false;
    }
  }
  return true;
}

template <concepts::Game Game>
void GameServer<Game>::GameSlot::handle_terminal(const ValueTensor& outcome) {
  ValueArray array = GameResults::to_value_array(outcome);
  for (auto player2 : players_) {
    player2->end_game(state_history_.current(), outcome);
  }

  TrainingDataWriter* training_data_writer = shared_data_.training_data_writer();
  if (training_data_writer) {
    game_log_->add_terminal(state_history_.current(), outcome);
    training_data_writer->add(game_log_);
  }

  if (params().announce_game_results) {
    std::stringstream ss;
    ss << std::format("Game {} complete.\n", game_id_);
    for (player_id_t p = 0; p < kNumPlayers; ++p) {
      ss << std::format("  pid={} name={} {}\n", p, players_[p]->get_name(), array[p]);
    }
    LOG_INFO("{}", ss.str());
  }

  // reindex outcome according to player_id
  ValueArray reindexed_outcome;
  for (int p = 0; p < kNumPlayers; ++p) {
    reindexed_outcome[player_order_[p].player_id] = array[p];
  }
  shared_data_.update(reindexed_outcome);

  game_started_ = false;
}

template <concepts::Game Game>
bool GameServer<Game>::GameSlot::start_game() {
  if (!shared_data_.request_game()) return false;

  game_id_ = util::get_unique_id();
  if (shared_data_.training_data_writer()) {
    game_log_ = std::make_shared<GameWriteLog>(game_id_, util::ns_since_epoch());
  }

  player_order_ = shared_data_.generate_player_order(instantiations_);

  for (int p = 0; p < kNumPlayers; ++p) {
    players_[p] = player_order_[p].player;
    player_names_[p] = players_[p]->get_name();
  }

  for (int p = 0; p < kNumPlayers; ++p) {
    players_[p]->init_game(game_id_, player_names_, p, game_log_);
    players_[p]->start_game();
  }

  if (params().mean_noisy_moves) {
    num_noisy_starting_moves_ = util::Random::exponential(1.0 / params().mean_noisy_moves);
  }
  game_started_ = true;

  state_history_.initialize(Rules{});
  move_number_ = 0;
  action_mode_ = -1;
  active_seat_ = -1;
  noisy_mode_ = false;
  mid_yield_ = false;
  pre_step();

  if (params().print_game_states) {
    Game::IO::print_state(std::cout, state_history_.current(), -1, &player_names_);
  }

  return true;
}

template <concepts::Game Game>
GameServer<Game>::GameThread::GameThread(SharedData& shared_data, game_thread_id_t id)
    : shared_data_(shared_data), id_(id) {
}

template <concepts::Game Game>
GameServer<Game>::GameThread::~GameThread() {
  join();
}

template <concepts::Game Game>
void GameServer<Game>::GameThread::join() {
  if (thread_.joinable()) thread_.join();
}

template <concepts::Game Game>
void GameServer<Game>::GameThread::launch() {
  thread_ = std::thread([&] { run(); });
}

template <concepts::Game Game>
void GameServer<Game>::GameThread::run() {
  shared_data_.increment_active_thread_count();
  while (true) {
    int64_t wait_for_game_slot_time_ns = 0;
    SlotContext item;
    if (!shared_data_.next(wait_for_game_slot_time_ns, item)) {
      if (shared_data_.paused()) {
        shared_data_.increment_paused_thread_count();
        shared_data_.wait_for_unpause();
        shared_data_.decrement_paused_thread_count();
        continue;
      } else {
        // no more slots to run, time to exit
        break;
      }
    }

    GameSlot* slot = shared_data_.get_game_slot(item.slot);

    // mcts_time_ns will include all other timed components of SearchThreadPerfStats, except for
    // wait_for_game_slot_time_ns. In PerfStats::calibrate(), we will undo this double-counting.
    int64_t mcts_time_ns = 0;
    core::PerfClocker clocker(mcts_time_ns);
    util::release_assert(slot->game_started());
    EnqueueRequest request = slot->step(item.context);

    bool dropped = false;
    if (slot->game_ended()) {
      util::release_assert(request.instruction == kEnqueueNow && request.extra_enqueue_count == 0);
      if (!slot->start_game()) {
        shared_data_.drop_slot();
        dropped = true;
      }
    }

    LOG_DEBUG("<-- GameServer::step(item={}:{}) enqueue_request={}:{} dropped={}", slot->id(),
              item.context, (int)request.instruction, request.extra_enqueue_count, dropped);

    if (!dropped) shared_data_.enqueue(item, request);

    clocker.stop();
    shared_data_.increment_mcts_time_ns(mcts_time_ns);
    shared_data_.increment_game_slot_time_ns(wait_for_game_slot_time_ns);
  }
  shared_data_.decrement_active_thread_count();
}

template <concepts::Game Game>
GameServer<Game>::GameServer(const Params& params,
                             const TrainingDataWriterParams& training_data_writer_params)
    : PerfStatsClient(), GameServerBase(), shared_data_(this, params, training_data_writer_params) {
  if (LoopControllerClient::initialized()) {
    LoopControllerClient* client = LoopControllerClient::get();
    client->add_listener(this);
  }
}

template <concepts::Game Game>
void GameServer<Game>::wait_for_remote_player_registrations() {
  util::clean_assert(num_registered_players() <= kNumPlayers,
                     "Invalid number of players registered: {}", num_registered_players());

  // fill in missing slots with remote players
  int shortage = kNumPlayers - num_registered_players();
  for (int i = 0; i < shortage; ++i) {
    RemotePlayerProxyGenerator* gen = new RemotePlayerProxyGenerator();
    shared_data_.register_player(-1, gen, true);
  }

  std::vector<PlayerRegistration*> remote_player_registrations;
  for (int r = 0; r < shared_data_.num_registrations(); ++r) {
    PlayerRegistration& reg = shared_data_.registration_templates()[r];
    if (dynamic_cast<RemotePlayerProxyGenerator*>(reg.gen)) {
      util::clean_assert(reg.seat < 0, "Cannot specify --seat= when using --type=Remote");
      remote_player_registrations.push_back(&reg);
    }
  }

  if (remote_player_registrations.empty()) return;

  int port = get_port();
  util::clean_assert(port > 0, "Invalid port number {}", port);

  io::Socket* server_socket = io::Socket::create_server_socket(port, kNumPlayers);

  LOG_INFO("Waiting for remote players to connect on port {}", port);
  int n = remote_player_registrations.size();
  int r = 0;
  while (r < n) {
    io::Socket* socket = server_socket->accept();

    int remaining_requests = 1;
    do {
      Packet<Registration> packet;
      if (!packet.read_from(socket)) {
        throw util::CleanException("Unexpected socket close");
      }
      const Registration& registration = packet.payload();
      std::string registered_name = registration.dynamic_size_section.player_name;
      remaining_requests = registration.remaining_requests;
      int max_simultaneous_games = registration.max_simultaneous_games;
      seat_index_t seat = registration.requested_seat;

      auto reg = remote_player_registrations[r++];
      if (!registered_name.empty()) {
        reg->gen->set_name(registered_name);
      }
      std::string name = reg->gen->get_name();
      util::clean_assert(!name.empty(), "Unexpected empty name");

      RemotePlayerProxyGenerator* gen = dynamic_cast<RemotePlayerProxyGenerator*>(reg->gen);
      gen->initialize(socket, max_simultaneous_games, reg->player_id);
      reg->seat = seat;

      Packet<RegistrationResponse> response_packet;
      response_packet.payload().player_id = reg->player_id;
      response_packet.set_player_name(name);
      response_packet.send_to(socket);

      LOG_INFO("Registered player: \"{}\" (seat: {}, remaining: {})\n", name.c_str(), seat,
               remaining_requests);
    } while (remaining_requests);
  }
}

template <concepts::Game Game>
void GameServer<Game>::run() {
  Game::static_init();
  wait_for_remote_player_registrations();
  shared_data_.init_random_seat_indices();
  util::clean_assert(shared_data_.ready_to_start(), "Game not ready to start");

  shared_data_.init_slots();
  shared_data_.run_yield_manager();
  create_threads();
  shared_data_.start_session();
  RemotePlayerProxy<Game>::PacketDispatcher::start_all(shared_data_.num_slots());
  shared_data_.start_games();

  time_point_t start_time = std::chrono::steady_clock::now();
  launch_threads();
  join_threads();
  time_point_t end_time = std::chrono::steady_clock::now();

  if (shared_data_.training_data_writer()) {
    shared_data_.training_data_writer()->wait_until_batch_empty();
  }

  int num_games = shared_data_.num_games_started();
  duration_t duration = end_time - start_time;
  int64_t ns = duration.count();

  results_array_t results = shared_data_.get_results();

  fprintf(stderr, "\n");  // flush progress-bar
  LOG_INFO("All games complete!");
  for (player_id_t p = 0; p < kNumPlayers; ++p) {
    LOG_INFO("pid={} name={} {}", p, shared_data_.get_player_name(p), get_results_str(results[p]));
  }

  util::KeyValueDumper::add("Parallelism factor", "%d", (int)threads_.size());
  util::KeyValueDumper::add("Num games", "%d", num_games);
  util::KeyValueDumper::add("Total runtime", "%.3fs", ns * 1e-9);
  util::KeyValueDumper::add("Avg runtime", "%.3fs", ns * 1e-9 / num_games);

  for (auto thread : threads_) {
    delete thread;
  }

  shared_data_.end_session();
  util::KeyValueDumper::flush();
}

template <concepts::Game Game>
void GameServer<Game>::create_threads() {
  int num_threads = params().num_game_threads;
  for (int t = 0; t < num_threads; ++t) {
    GameThread* thread = new GameThread(shared_data_, t);
    threads_.push_back(thread);
  }
  core::Globals::num_game_threads = num_threads;
}

template <concepts::Game Game>
void GameServer<Game>::launch_threads() {
  for (auto thread : threads_) {
    thread->launch();
  }
}

template <concepts::Game Game>
void GameServer<Game>::join_threads() {
  for (auto thread : threads_) {
    thread->join();
  }
}

template <concepts::Game Game>
void GameServer<Game>::update_perf_stats(PerfStats& stats) {
  shared_data_.update_perf_stats(stats);
}

template <concepts::Game Game>
std::string GameServer<Game>::get_results_str(const results_map_t& map) {
  int win = 0;
  int loss = 0;
  int draw = 0;
  float score = 0;

  for (auto it : map) {
    float f = it.first;
    int count = it.second;
    score += f * count;
    if (f == 1)
      win += count;
    else if (f == 0)
      loss += count;
    else
      draw += count;
  }
  return std::format("W{} L{} D{} [{:.16g}]", win, loss, draw, score);;
}

}  // namespace core
