#include "core/GameServer.hpp"

#include "core/BasicTypes.hpp"
#include "core/LoopControllerClient.hpp"
#include "core/Packet.hpp"
#include "core/PerfStats.hpp"
#include "core/players/RemotePlayerProxy.hpp"
#include "generic_players/AnalysisPlayerGenerator.hpp"
#include "util/BoostUtil.hpp"
#include "util/CompactBitSet.hpp"
#include "util/CppUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/KeyValueDumper.hpp"
#include "util/LoggingUtil.hpp"
#include "util/Random.hpp"
#include "util/Rendering.hpp"
#include "util/SocketUtil.hpp"
#include "util/StringUtil.hpp"

#include <arpa/inet.h>
#include <boost/program_options.hpp>
#include <magic_enum/magic_enum.hpp>
#include <magic_enum/magic_enum_format.hpp>
#include <netinet/in.h>
#include <sys/socket.h>

#include <format>
#include <iostream>
#include <string>
#include <vector>

namespace core {

namespace detail {

/*
 * Between machine reboots, no two calls to this function from the same machine should return equal
 * values.
 */
inline int64_t get_unique_game_id() {
  static mit::mutex mut;
  mit::lock_guard<mit::mutex> lock(mut);

  static int64_t last = 0;
  int64_t id = util::ns_since_epoch();
  while (id <= last) {
    id = util::ns_since_epoch();
  }
  last = id;
  return id;
}

}  // namespace detail

template <concepts::Game Game>
auto GameServer<Game>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("GameServer options");

  return desc
    .template add_option<"port">(po::value<int>(&port)->default_value(port),
                                 "port for external players to connect to (must be set to a "
                                 "nonzero value if using external players)")
    .template add_option<"initial-actions">(
      po::value<std::string>(&initial_actions_str)->default_value(initial_actions_str),
      "initial actions of each game (comma-separated integers, e.g. \"0,1,2\")")
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
      "ignore imminent victory claims from players")
    .template add_option<"alternating-mode">(
      po::value<int>(&alternating_mode)->default_value(alternating_mode),
      "alternating mode (0: disable, 1: auto-enable, 2: enable)")
    .template add_option<"analysis-mode">(
      boost::program_options::bool_switch(&analysis_mode)->default_value(analysis_mode),
      "enable analysis mode where the stepping of a game is controlled externally");
}

template <concepts::Game Game>
GameServer<Game>::SharedData::SharedData(GameServer* server, const Params& params)
    : server_(server), params_(params), yield_manager_(cv_, mutex_, queue_, pending_queue_count_) {
  if (params_.alternating_mode == 0) {
    global_active_player_id_ = -1;
  } else if (params_.alternating_mode == 1) {
    // auto-enable - can change later
    global_active_player_id_ = -1;
  } else if (params_.alternating_mode == 2) {
    global_active_player_id_ = 0;
  } else {
    throw util::CleanException("GameServer::{}(): invalid alternating mode: {}", __func__,
                               params_.alternating_mode);
  }
}

template <concepts::Game Game>
GameServer<Game>::SharedData::~SharedData() {
  if (state_thread_.joinable()) {
    state_thread_.join();
  }
  if (bar_) delete bar_;

  for (auto& reg : registrations_) {
    delete reg.gen;
  }

  for (GameSlot* slot : game_slots_) {
    delete slot;
  }
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

    if (global_active_player_id_ >= 0) {
      // In alternating mode, we only add the slot to the queue if it's the active player's turn
      if (slot->active_player_id() == global_active_player_id_) {
        queue_.emplace(slot->id());
      } else {
        deferred_queues_[slot->active_player_id()].emplace(slot->id());
        deferred_count_++;
        pending_queue_count_++;
      }
    } else {  // not in alternating mode
      queue_.emplace(slot->id());
    }
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::init_progress_bar() {
  mit::lock_guard<mit::mutex> guard(mutex_);
  if (bar_) return;

  if (params_.display_progress_bar && params_.num_games > 0 &&
      util::Rendering::mode() == util::Rendering::kTerminal) {
    bar_ = new progressbar(params_.num_games + 1);  // + 1 for first update
    bar_->update();                                 // so that progress-bar displays immediately
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::init_random_seat_indices() {
  util::CompactBitSet<kNumPlayers> fixed_seat_indices;
  for (PlayerRegistration& reg : registrations_) {
    if (reg.seat >= 0) {
      fixed_seat_indices.set(reg.seat);
    }
  }

  for (seat_index_t random_seat : fixed_seat_indices.off_indices()) {
    random_seat_indices_[num_random_seats_++] = random_seat;
  }

  util::Random::shuffle(&random_seat_indices_[0], &random_seat_indices_[num_random_seats_]);
}

template <concepts::Game Game>
GameServerBase::next_result_t GameServer<Game>::SharedData::next(
  int64_t& wait_for_game_slot_time_ns, SlotContext& item) {
  core::PerfClocker clocker(wait_for_game_slot_time_ns);
  mit::unique_lock lock(mutex_);

  if (queue_.empty()) {
    if (pending_queue_count_ > 0) {
      // NOTE: we need to unlock the mutex before calling force_progress() to avoid a deadlock
      // within NNEvaluationService
      lock.unlock();
      server_->force_progress();
      lock.lock();
      waiting_in_next_ = true;
      cv_.wait(lock, [&] {
        if (pause_state_ != kUnpaused) return true;
        if (!queue_.empty()) return true;
        if (pending_queue_count_ == 0) return true;
        if (deferred_count_ == pending_queue_count_) return true;
        return false;
      });
      waiting_in_next_ = false;
    }
    if (queue_.empty()) {
      if (pending_queue_count_ == 0) {
        LOG_DEBUG("<-- GameServer::{}(): exiting", __func__);
        return kExit;
      }

      if (deferred_count_ == pending_queue_count_) {
        DEBUG_ASSERT(global_active_player_id_ >= 0,
                     "GameServer::{}(): deferred_count_ == pending_queue_count_ ({} == {}) but "
                     "global_active_player_id_ is not set",
                     __func__, deferred_count_, pending_queue_count_);

        DEBUG_ASSERT(deferred_queues_[global_active_player_id_].empty(),
                     "GameServer::{}(): deferred queue for active player {} is not empty", __func__,
                     global_active_player_id_);

        increment_global_active_player_id();
        queue_.swap(deferred_queues_[global_active_player_id_]);
        deferred_count_ -= queue_.size();
        pending_queue_count_ -= queue_.size();
        validate_deferred_count();
        LOG_DEBUG(
          "<-- GameServer::{}(): handle deferral (global_active_player_id_:{} deferred_count_:{} "
          "pending_queue_count_:{} queue_.size():{})",
          __func__, global_active_player_id_, deferred_count_, pending_queue_count_, queue_.size());

        lock.unlock();
        return next(wait_for_game_slot_time_ns, item);
      }
    }
  }

  if (pause_state_ != kUnpaused) {
    LOG_DEBUG("<-- GameServer::{}(): pause_state={}", __func__, pause_state_);
    return kHandlePause;
  }

  DEBUG_ASSERT(!queue_.empty(), "GameServer::{}(): queue should not be empty here", __func__);

  // TODO: there is a potential weird race condition on kYield responses. For example, with
  // WebPlayer, we can have the following sequence:
  //
  // 1. WebPlayer::get_action_response() is called, which notifies its response loop and returns
  //    ActionResponse::yield().
  // 2. Before the yield is returned, the following events happen:
  //    A. The response loop is woken up, and the notification unit is notified.
  //    B. The notification causes the GameSlot to be put back into the GameServer queue.
  //    C. GameServer::next() is called, which pops that item from the queue.
  //
  // This would result in the second next() call "lapping" the first one. I think this would violate
  // some assumptions in this logic.
  //
  // One solution is to add "yield callback" mechanics, but this would complexify the player
  // interface. I think it would be better to add appropriate checks in the GameServer logic to
  // detect this situation and handle it gracefully.
  //
  // This race condition would be virtually impossible to trigger in practice, but with the mit
  // library, we should be able to force it during a unit test.
  item = queue_.front();
  queue_.pop();
  pending_queue_count_++;
  LOG_DEBUG("<-- GameServer::{}(): item={}:{} (queue:{} pending:{})", __func__, item.slot,
            item.context, queue_.size(), pending_queue_count_);

  DEBUG_ASSERT(global_active_player_id_ < 0 ||
                 game_slots_[item.slot]->active_player_id() == global_active_player_id_,
               "GameServer::{}(): item's active player id ({}) does not match "
               "global_active_player_id_ ({})",
               __func__, game_slots_[item.slot]->active_player_id(), global_active_player_id_);

  return kProceed;
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::enqueue(SlotContext item, const EnqueueRequest& request) {
  mit::unique_lock lock(mutex_);
  auto& queue = get_queue_to_use(item.slot);
  bool deferred = &queue != &queue_;
  if (request.instruction == kEnqueueNow) {
    RELEASE_ASSERT(request.extra_enqueue_count == 0);
    item.context = 0;  // when continuing, we always want to reset the context to 0
    queue.push(item);
    pending_queue_count_ -= !deferred;
    deferred_count_ += deferred;
  } else if (request.instruction == kEnqueueLater) {
    if (request.extra_enqueue_count > 0) {
      // Push back the item's siblings
      for (int i = 0; i < request.extra_enqueue_count; ++i) {
        item.context = i + 1;
        queue.push(item);
      }
      if (deferred) {
        pending_queue_count_ += request.extra_enqueue_count;
        deferred_count_ += request.extra_enqueue_count;
      }
      item.context = 0;  // just for the LOG_DEBUG() statement below
    }
  } else if (request.instruction == kEnqueueNever) {
    pending_queue_count_--;
  } else {
    throw util::Exception("GameServer::{}(): invalid enqueue instruction: {}", __func__,
                          request.instruction);
  }

  LOG_DEBUG("<-- GameServer::{}(item={}:{}, request={}:{}) pending={} deferred={}", __func__,
            item.slot, item.context, request.instruction, request.extra_enqueue_count,
            pending_queue_count_, deferred_count_);
  validate_deferred_count();

  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
bool GameServer<Game>::SharedData::request_game() {
  if (LoopControllerClient::deactivated()) return false;

  mit::lock_guard<mit::mutex> guard(mutex_);
  if (params_.num_games > 0 && num_games_started_ >= params_.num_games) return false;
  num_games_started_++;
  return true;
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::update(const ValueArray& outcome) {
  mit::lock_guard<mit::mutex> guard(mutex_);
  num_games_ended_++;
  for (seat_index_t s = 0; s < kNumPlayers; ++s) {
    results_array_[s][outcome[s]]++;
  }

  if (bar_) bar_->update();
}

template <concepts::Game Game>
auto GameServer<Game>::SharedData::get_results() const {
  mit::lock_guard<mit::mutex> guard(mutex_);
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
  if (params_.analysis_mode) {
    gen = new generic::AnalysisPlayerGenerator<Game>(gen);
  }

  CLEAN_ASSERT(seat < kNumPlayers, "Invalid seat number {}", seat);
  if (dynamic_cast<RemotePlayerProxyGenerator*>(gen)) {
    if (implicit_remote) {
      CLEAN_ASSERT(
        params_.port > 0,
        "If specifying fewer than {} --player's, the remaining players are assumed to be remote "
        "players. In this case, --port must be specified, so that the remote players can connect",
        kNumPlayers);
    } else {
      CLEAN_ASSERT(params_.port > 0, "Cannot use remote players without setting --port");
    }

    CLEAN_ASSERT(seat < 0, "Cannot specify --seat with --type=Remote");
  }

  if (seat >= 0) {
    for (const auto& reg : registrations_) {
      CLEAN_ASSERT(reg.seat != seat, "Double-seated player at seat {}", seat);
    }
  }
  player_id_t player_id = registrations_.size();
  CLEAN_ASSERT(player_id < kNumPlayers, "Too many players registered (max {})", kNumPlayers);
  std::string name = gen->get_name();
  if (name.empty()) {
    gen->set_name(gen->get_default_name());
  }
  registrations_.emplace_back(gen, seat, player_id);

  if (gen->supports_backtracking()) {
    backtracking_support_.set(player_id);
  }
}

template <concepts::Game Game>
typename GameServer<Game>::player_instantiation_array_t
GameServer<Game>::SharedData::generate_player_order(
  const player_instantiation_array_t& instantiations) {
  mit::unique_lock lock(mutex_);
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
void GameServer<Game>::SharedData::handle_alternating_mode_recommendation() {
  if (params_.alternating_mode == 1) {  // auto-enable
    global_active_player_id_ = 0;
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::debug_dump() const {
  mit::unique_lock lock(mutex_);
  LOG_WARN(
    "GameServer {} pause_state:{} queue.size():{} pending_queue_count:{} deferred_count_:{} "
    "active_thread_count:{} paused_thread_count:{} waiting_in_next:{} num_games_started:{} "
    "num_games_ended:{}",
    __func__, pause_state_, queue_.size(), pending_queue_count_, deferred_count_,
    active_thread_count_, paused_thread_count_, waiting_in_next_, num_games_started_,
    num_games_ended_);

  for (int i = 0; i < (int)game_slots_.size(); ++i) {
    GameSlot* slot = game_slots_[i];
    bool mid_yield = slot->mid_yield();
    bool continue_hit = slot->continue_hit();
    bool in_critical_section = slot->in_critical_section();

    if (mid_yield || continue_hit || in_critical_section) {
      std::ostringstream ss;
      Game::IO::print_state(ss, slot->state());

      Player* player = slot->active_player();
      LOG_WARN(
        "GameServer {} game_slot[{}] mid_yield:{} continue_hit:{} in_critical_section:{} "
        "active_seat:{} active_player:{} state:\n{}",
        __func__, i, mid_yield, continue_hit, in_critical_section, slot->active_seat(),
        player ? player->get_name() : "-", ss.str());
    }
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::pause() {
  LOG_INFO("GameServer: pausing");
  mit::unique_lock lock(mutex_);
  cv_.wait(lock, [&] { return pause_state_ == kUnpaused && in_prelude_count_ == 0; });
  pause_state_ = kPausing;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::unpause() {
  LOG_INFO("GameServer: unpausing");
  mit::unique_lock lock(mutex_);
  RELEASE_ASSERT(pause_state_ == kPaused, "{}(): {} != {} @{}", __func__, pause_state_, kPaused,
                 __LINE__);
  pause_state_ = kUnpausing;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::increment_active_thread_count() {
  mit::unique_lock lock(mutex_);
  active_thread_count_++;

  bool ready_to_launch_state_thread = active_thread_count_ == num_initial_threads_;
  if (ready_to_launch_state_thread) {
    state_thread_launched_ = true;
    state_thread_ = mit::thread([&] { this->state_loop(); });
    lock.unlock();
    cv_.notify_all();
  } else {
    cv_.wait(lock, [&] { return state_thread_launched_; });
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::decrement_active_thread_count() {
  mit::unique_lock lock(mutex_);
  active_thread_count_--;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::run_prelude(core::game_thread_id_t id) {
  if (pause_state_ == kUnpaused) return;  // avoid mutex in common case

  mit::unique_lock lock(mutex_);
  in_prelude_count_++;

  RELEASE_ASSERT(pause_state_ == kPausing, "{}(): {} != {} @{}", __func__, pause_state_, kPausing,
                 __LINE__);
  paused_thread_count_++;
  bool notify = paused_thread_count_ == active_thread_count_;
  LOG_INFO("<-- GameServer: thread={} pause={} in_prelude={} active={} notify={} @{}", id,
           paused_thread_count_, in_prelude_count_, active_thread_count_, notify, __LINE__);
  if (notify) {
    lock.unlock();
    cv_.notify_all();
    lock.lock();
  }

  cv_.wait(lock, [&] {
    if (pause_state_ == kUnpausing) return true;
    LOG_DEBUG("<-- GameServer: thread {} still waiting for pause state {} (current: {})", id,
              kUnpausing, pause_state_);
    return false;
  });

  paused_thread_count_--;
  notify = paused_thread_count_ == 0;
  LOG_INFO("<-- GameServer: thread={} pause={} in_prelude={} active={} notify={} @{}", id,
           paused_thread_count_, in_prelude_count_, active_thread_count_, notify, __LINE__);
  if (notify) {
    lock.unlock();
    cv_.notify_all();
    lock.lock();
  }

  cv_.wait(lock, [&] {
    if (pause_state_ == kUnpaused) return true;
    LOG_DEBUG("<-- GameServer: thread {} still waiting for pause state {} (current: {})", id,
              kUnpaused, pause_state_);
    return false;
  });

  in_prelude_count_--;
  notify = in_prelude_count_ == 0;
  LOG_INFO("<-- GameServer: thread={} pause={} in_prelude={} active={} notify={} @{}", id,
           paused_thread_count_, in_prelude_count_, active_thread_count_, notify, __LINE__);
  if (notify) {
    lock.unlock();
    cv_.notify_all();
  }
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::state_loop() {
  mit::unique_lock lock(mutex_);
  while (true) {
    cv_.wait(lock, [&] { return active_thread_count_ == 0 || pause_state_ != kUnpaused; });
    if (active_thread_count_ == 0) break;

    RELEASE_ASSERT(pause_state_ == kPausing, "{}(): {} != {} @{}", __func__, pause_state_, kPausing,
                   __LINE__);

    LOG_INFO("GameServer: pausing, waiting for threads to be ready for pause...");
    cv_.wait(lock, [&] {
      // Wait until all threads are paused
      return paused_thread_count_ == active_thread_count_;
    });

    LOG_INFO("GameServer: all threads ready for pause, issuing pause receipt...");
    pause_state_ = kPaused;
    core::LoopControllerClient::get()->handle_pause_receipt(__FILE__, __LINE__);

    LOG_INFO("GameServer: paused, waiting for unpause from loop controller...");
    cv_.wait(lock, [&] { return active_thread_count_ == 0 || pause_state_ != kPaused; });
    if (active_thread_count_ == 0) break;

    RELEASE_ASSERT(pause_state_ == kUnpausing, "{}(): {} != {} @{}", __func__, pause_state_,
                   kUnpausing, __LINE__);

    LOG_INFO("GameServer: unpausing, waiting for threads to be ready for unpause...");
    cv_.wait(lock, [&] {
      // Wait until all threads are unpaused
      return paused_thread_count_ == 0;
    });

    pause_state_ = kUnpaused;
    core::LoopControllerClient::get()->handle_unpause_receipt(__FILE__, __LINE__);
    LOG_INFO("GameServer: unpaused!");
    lock.unlock();
    cv_.notify_all();
    lock.lock();
  }
}

template <concepts::Game Game>
slot_context_queue_t& GameServer<Game>::SharedData::get_queue_to_use(game_slot_index_t g) {
  if (global_active_player_id_ < 0) {
    // Not in alternating mode, use the main queue
    return queue_;
  }

  GameSlot* slot = game_slots_[g];
  player_id_t p = slot->active_player_id();
  if (p == global_active_player_id_) {
    // This slot is for the active player, use the main queue
    return queue_;
  }
  // This slot is for a deferred player, use the deferred queue for that player
  return deferred_queues_[p];
}

template <concepts::Game Game>
void GameServer<Game>::SharedData::validate_deferred_count() const {
  // assumes mutex_ is locked
  if (!IS_DEFINED(DEBUG_BUILD)) return;

  int deferred_count = 0;
  for (const auto& queue : deferred_queues_) {
    deferred_count += queue.size();
  }
  DEBUG_ASSERT(deferred_count == deferred_count_,
               "GameServer::{}(): deferred_count_ mismatch: {} != {}", __func__, deferred_count_,
               deferred_count);
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
    : shared_data_(shared_data), id_(id) {
  bool disable_progress_bar = false;

  for (int p = 0; p < kNumPlayers; ++p) {
    PlayerRegistration& reg = shared_data_.registration_templates()[p];
    instantiations_[p] = reg.instantiate(id);
    disable_progress_bar |= instantiations_[p].player->disable_progress_bar();
  }

  int num_backtracking_supporting_players = shared_data_.backtracking_support().count();
  if (num_backtracking_supporting_players >= 2) {
    for (auto& inst : instantiations_) {
      inst.player->set_facing_backtracking_opponent();
    }
  } else if (num_backtracking_supporting_players == 1) {
    for (auto ix : shared_data_.backtracking_support().off_indices()) {
      instantiations_[ix].player->set_facing_backtracking_opponent();
    }
  }

  if (!disable_progress_bar) {
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
GameServerBase::StepResult GameServer<Game>::GameSlot::step(context_id_t context) {
  StepResult result;
  if (!Rules::is_chance_mode(action_mode_)) {
    if (step_non_chance(context, result)) {
      CriticalSectionCheck check(in_critical_section_);
      pre_step();
    } else {
      return result;
    }
  }

  if (Rules::is_chance_mode(action_mode_)) {
    if (step_chance(result)) {
      CriticalSectionCheck check(in_critical_section_);
      pre_step();
    }
  }
  return result;
}

template <concepts::Game Game>
void GameServer<Game>::GameSlot::pre_step() {
  DEBUG_ASSERT(!mid_yield_);

  // Even with multi-threading enabled via ActionResponse::extra_enqueue_count, we should never
  // get here with multiple threads

  chance_action_ = -1;
  action_mode_ = Rules::get_action_mode(state());
  noisy_mode_ = move_number_ < num_noisy_starting_moves_;
  if (!Rules::is_chance_mode(action_mode_)) {
    active_seat_ = Rules::get_current_player(state());
    valid_actions_ = Rules::get_legal_moves(state());
  }
}

template <concepts::Game Game>
bool GameServer<Game>::GameSlot::step_chance(StepResult& result) {
  if (chance_action_ < 0) {
    ChanceDistribution chance_dist = Rules::get_chance_distribution(state());
    chance_action_ = eigen_util::sample(chance_dist);
    apply_action(chance_action_);
  }

  EnqueueRequest& enqueue_request = result.enqueue_request;
  for (; step_chance_player_index_ < kNumPlayers; ++step_chance_player_index_) {
    Player* player = players_[step_chance_player_index_];
    YieldNotificationUnit notification_unit(shared_data_.yield_manager(), id_, 0);
    ChanceEventHandleRequest request(notification_unit, state(), chance_action_);

    core::yield_instruction_t response = player->handle_chance_event(request);

    switch (response) {
      case kContinue: {
        mid_yield_ = false;
        break;
      }
      case kYield: {
        mid_yield_ = true;
        enqueue_request.instruction = kEnqueueLater;
        return false;
      }
      default: {
        throw util::Exception("Unexpected response: {}", response);
      }
    }
  }

  CriticalSectionCheck check(in_critical_section_);

  step_chance_player_index_ = 0;  // reset for next chance event

  if (params().print_game_states) {
    Game::IO::print_state(std::cout, state(), chance_action_, &player_names_);
  }

  StateChangeUpdate update(active_seat_, state(), chance_action_, state_node_index_, action_mode_);
  for (auto player2 : players_) {
    player2->receive_state_change(update);
  }

  GameResultTensor outcome;
  if (Game::Rules::is_terminal(state(), active_seat_, chance_action_, outcome)) {
    handle_terminal(outcome, result);
    return false;
  }
  return true;
}

template <concepts::Game Game>
bool GameServer<Game>::GameSlot::step_non_chance(context_id_t context, StepResult& result) {
  Player* player = players_[active_seat_];
  YieldNotificationUnit notification_unit(shared_data_.yield_manager(), id_, context);
  ActionRequest request(state(), valid_actions_, notification_unit, get_player_aux());
  request.play_noisily = noisy_mode_;
  request.undo_allowed = undo_allowed();

  ActionResponse response = player->get_action_response(request);
  int extra_enqueue_count = response.get_extra_enqueue_count();
  yield_instruction_t yield_instr = response.get_yield_instruction();
  DEBUG_ASSERT(extra_enqueue_count == 0 || yield_instr == kYield,
               "Invalid response: extra={} instr={}", extra_enqueue_count,
               int(yield_instr));

  EnqueueRequest& enqueue_request = result.enqueue_request;

  if (yield_instr == kContinue) {
    CriticalSectionCheck check(in_critical_section_);
    mid_yield_ = false;
    continue_hit_ = true;
  }

  RELEASE_ASSERT(request.permits(response), "ActionResponse {} not permitted by ActionRequest",
                 response.type());
  switch (response.type()) {
    case ActionResponse::kMakeMove:
      break;

    case ActionResponse::kUndoLastMove:
      undo_player_last_action();
      // TODO: propagate backtrack to players. Today we only rewind the server's state_node_index_.
      // Players that maintain internal search/UI history may become inconsistent (e.g.
      // alpha0::Player). The right mechanism is likely an explicit "backtrack" notification or a
      // full state resync, which depends on how we factor Player/Manager.
      return true;

    case ActionResponse::kBacktrack:
      throw util::CleanException("BackTrack not yet implemented in GameServer");

    case ActionResponse::kResignGame:
      resign_game(result);
      return false;

    case ActionResponse::kYieldResponse:
      RELEASE_ASSERT(!continue_hit_, "kYield after continue hit!");
      mid_yield_ = true;
      enqueue_request.instruction = kEnqueueLater;
      enqueue_request.extra_enqueue_count = extra_enqueue_count;
      return false;

    case ActionResponse::kDropResponse:
      enqueue_request.instruction = kEnqueueNever;
      return false;

    default:
      throw util::Exception("Unexpected ActionResponse type: {}", response.type());
  }

  if (response.is_aux_set()) {
    set_player_aux(response.aux());
  }

  CriticalSectionCheck check2(in_critical_section_);
  RELEASE_ASSERT(!mid_yield_);

  continue_hit_ = false;
  move_number_++;
  action_t action = response.get_action();

  if (response.get_victory_guarantee() && params().respect_victory_hints) {
    GameResultTensor outcome = GameResults::win(active_seat_);
    if (params().announce_game_results) {
      LOG_INFO("Short-circuiting game {} because player {} (seat={}) claims victory", game_id_,
               player->get_name(), active_seat_);
    }
    handle_terminal(outcome, result);
    return false;
  } else {
    // TODO: gracefully handle and prompt for retry. Otherwise, a malicious remote process can crash
    // the server.
    RELEASE_ASSERT(valid_actions_[action], "Invalid action: {}", action);

    apply_action(action);
    if (params().print_game_states) {
      Game::IO::print_state(std::cout, state(), action, &player_names_);
    }

    StateChangeUpdate update(active_seat_, state(), action, state_node_index_, action_mode_);
    for (auto player2 : players_) {
      player2->receive_state_change(update);
    }

    GameResultTensor outcome;
    if (Game::Rules::is_terminal(state(), active_seat_, action, outcome)) {
      handle_terminal(outcome, result);
      return false;
    }
  }
  return true;
}

template <concepts::Game Game>
void GameServer<Game>::GameSlot::handle_terminal(const GameResultTensor& outcome,
                                                 StepResult& result) {
  ValueArray array = GameResults::to_value_array(outcome);
  for (auto player2 : players_) {
    player2->end_game(state(), outcome);
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

  EnqueueRequest& request = result.enqueue_request;
  RELEASE_ASSERT(request.instruction == kEnqueueNow && request.extra_enqueue_count == 0);

  if (!start_game()) {
    request.instruction = kEnqueueNever;
  }

  result.game_ended = true;
}

template <concepts::Game Game>
bool GameServer<Game>::GameSlot::start_game() {
  if (!shared_data_.request_game()) return false;

  game_id_ = detail::get_unique_game_id();
  player_order_ = shared_data_.generate_player_order(instantiations_);

  for (int p = 0; p < kNumPlayers; ++p) {
    players_[p] = player_order_[p].player;
    player_names_[p] = players_[p]->get_name();
  }

  for (int p = 0; p < kNumPlayers; ++p) {
    players_[p]->init_game(game_id_, player_names_, p);
    if (!players_[p]->start_game()) return false;
  }

  if (params().mean_noisy_moves) {
    num_noisy_starting_moves_ = util::Random::exponential(1.0 / params().mean_noisy_moves);
  }
  game_started_ = true;

  move_number_ = 0;
  action_mode_ = -1;
  active_seat_ = -1;
  noisy_mode_ = false;
  mid_yield_ = false;

  state_tree_.init();
  state_node_index_ = 0;
  for (const core::action_t& action : shared_data_.initial_actions()) {
    pre_step();
    apply_action(action);

    StateChangeUpdate update(active_seat_, state(), action, state_node_index_, action_mode_);
    for (int p = 0; p < kNumPlayers; ++p) {
      players_[p]->receive_state_change(update);
    }
  }

  pre_step();

  if (params().print_game_states) {
    Game::IO::print_state(std::cout, state(), -1, &player_names_);
  }

  return true;
}

template <concepts::Game Game>
GameServer<Game>::GameThread::GameThread(SharedData& shared_data, game_thread_id_t id)
    : shared_data_(shared_data), id_(id) {}

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
  thread_ = mit::thread([&] { run(); });
}

template <concepts::Game Game>
void GameServer<Game>::GameThread::run() {
  shared_data_.increment_active_thread_count();
  while (true) {
    shared_data_.run_prelude(id_);
    int64_t wait_for_game_slot_time_ns = 0;
    SlotContext item;
    next_result_t next_result = shared_data_.next(wait_for_game_slot_time_ns, item);
    if (next_result == kExit) {
      break;
    } else if (next_result == kHandlePause) {
      continue;
    } else if (next_result == kProceed) {
      // do nothing
    } else {
      throw util::Exception("Invalid next_result: {}", next_result);
    }

    GameSlot* slot = shared_data_.get_game_slot(item.slot);

    LOG_DEBUG("<-- GameServer::step(item={}:{})", slot->id(), item.context);

    // mcts_time_ns will include all other timed components of SearchThreadPerfStats, except for
    // wait_for_game_slot_time_ns. In PerfStats::calibrate(), we will undo this double-counting.
    int64_t mcts_time_ns = 0;
    core::PerfClocker clocker(mcts_time_ns);
    RELEASE_ASSERT(slot->game_started());
    StepResult step_result = slot->step(item.context);
    EnqueueRequest& request = step_result.enqueue_request;

    LOG_DEBUG("<-- GameServer::step(item={}:{}) complete enqueue_request={}:{}", slot->id(),
              item.context, request.instruction, request.extra_enqueue_count);

    shared_data_.enqueue(item, request);

    clocker.stop();
    shared_data_.increment_mcts_time_ns(mcts_time_ns);
    shared_data_.increment_game_slot_time_ns(wait_for_game_slot_time_ns);
  }
  shared_data_.decrement_active_thread_count();
}

template <concepts::Game Game>
void GameServer<Game>::handle_alternating_mode_recommendation() {
  shared_data_.handle_alternating_mode_recommendation();
}

template <concepts::Game Game>
GameServer<Game>::GameServer(const Params& params)
    : PerfStatsClient(), GameServerBase(params.num_game_threads), shared_data_(this, params) {
  if (LoopControllerClient::initialized()) {
    LoopControllerClient* client = LoopControllerClient::get();
    client->add_listener(this);
  }

  if (!params.initial_actions_str.empty()) {
    std::vector<std::string> action_strs = util::split(params.initial_actions_str, ",");
    std::vector<core::action_t> initial_actions;
    for (const auto& action_str : action_strs) {
      core::action_t action = std::stoi(action_str);
      CLEAN_ASSERT(action >= 0, "Invalid initial action: {} in {}", action_str,
                   params.initial_actions_str);
      initial_actions.push_back(action);
    }
    set_initial_actions(initial_actions);
  }
}

template <concepts::Game Game>
void GameServer<Game>::wait_for_remote_player_registrations() {
  CLEAN_ASSERT(num_registered_players() <= kNumPlayers, "Invalid number of players registered: {}",
               num_registered_players());

  // fill in missing slots with remote players
  int shortage = kNumPlayers - num_registered_players();
  for (int i = 0; i < shortage; ++i) {
    RemotePlayerProxyGenerator* gen = new RemotePlayerProxyGenerator(this);
    shared_data_.register_player(-1, gen, true);
  }

  std::vector<PlayerRegistration*> remote_player_registrations;
  for (int r = 0; r < shared_data_.num_registrations(); ++r) {
    PlayerRegistration& reg = shared_data_.registration_templates()[r];
    if (dynamic_cast<RemotePlayerProxyGenerator*>(reg.gen)) {
      CLEAN_ASSERT(reg.seat < 0, "Cannot specify --seat= when using --type=Remote");
      remote_player_registrations.push_back(&reg);
    }
  }

  if (remote_player_registrations.empty()) return;

  int port = get_port();
  CLEAN_ASSERT(port > 0, "Invalid port number {}", port);

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
      CLEAN_ASSERT(!name.empty(), "Unexpected empty name");

      RemotePlayerProxyGenerator* gen = dynamic_cast<RemotePlayerProxyGenerator*>(reg->gen);
      gen->initialize(socket, max_simultaneous_games, reg->player_id);
      reg->seat = seat;

      Packet<RegistrationResponse> response_packet;
      response_packet.payload().player_id = reg->player_id;
      response_packet.set_player_name(name);
      response_packet.send_to(socket);

      LOG_INFO("Registered player: \"{}\" (seat: {}, remaining: {})", name.c_str(), seat,
               remaining_requests);
    } while (remaining_requests);
  }
}

template <concepts::Game Game>
void GameServer<Game>::setup() {
  Game::static_init();
  wait_for_remote_player_registrations();
  shared_data_.init_random_seat_indices();
  CLEAN_ASSERT(shared_data_.ready_to_start(), "Game not ready to start");

  shared_data_.init_slots();
  create_threads();
  shared_data_.start_session();
  RemotePlayerProxy<Game>::PacketDispatcher::start_all(shared_data_.num_slots());
  shared_data_.start_games();
}

template <concepts::Game Game>
void GameServer<Game>::run() {
  setup();

  time_point_t start_time = std::chrono::steady_clock::now();
  LOG_DEBUG("GameServer> Launching threads...");
  launch_threads();
  join_threads();
  time_point_t end_time = std::chrono::steady_clock::now();

  int num_games = shared_data_.num_games_started();
  duration_t duration = end_time - start_time;
  int64_t ns = duration.count();

  if (shared_data_.params().display_progress_bar) {
    fprintf(stderr, "\n");  // flush progress-bar
  }

  util::KeyValueDumper::add("Parallelism factor", "%d", (int)threads_.size());
  util::KeyValueDumper::add("Num games", "%d", num_games);
  util::KeyValueDumper::add("Total runtime", "%.3fs", ns * 1e-9);
  util::KeyValueDumper::add("Avg runtime", "%.3fs", ns * 1e-9 / num_games);

  for (auto thread : threads_) {
    delete thread;
  }

  shared_data_.end_session();
}

template <concepts::Game Game>
void GameServer<Game>::print_summary() const {
  results_array_t results = shared_data_.get_results();
  LOG_INFO("All games complete!");
  for (player_id_t p = 0; p < kNumPlayers; ++p) {
    LOG_INFO("pid={} name={} {}", p, shared_data_.get_player_name(p), get_results_str(results[p]));
  }

  util::KeyValueDumper::flush();
}

template <concepts::Game Game>
void GameServer<Game>::create_threads() {
  for (int t = 0; t < this->num_game_threads(); ++t) {
    GameThread* thread = new GameThread(shared_data_, t);
    threads_.push_back(thread);
  }
}

template <concepts::Game Game>
void GameServer<Game>::launch_threads() {
  shared_data_.set_num_initial_threads(threads_.size());
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
  return std::format("W{} L{} D{} [{:.16g}]", win, loss, draw, score);
  ;
}

template <concepts::Game Game>
bool GameServer<Game>::GameSlot::active_player_supports_backtracking() const {
  player_id_t player_id = player_order_[active_seat_].player_id;
  return shared_data_.backtracking_support()[player_id];
}

template <concepts::Game Game>
game_tree_index_t GameServer<Game>::GameSlot::player_last_action_node_index() const {

  for (auto ix = state_tree_.get_parent_index(state_node_index_); ix != kNullNodeIx;
       ix = state_tree_.get_parent_index(ix)) {

    bool is_current_player = state_tree_.get_active_seat(ix) == active_seat_;
    bool is_chance = state_tree_.is_chance_node(ix);

    if (is_current_player && !is_chance) {
      return ix;
    }
  }
  throw util::Exception("No previous action found for player {}", active_seat_);
}

template <concepts::Game Game>
void GameServer<Game>::GameSlot::resign_game(StepResult& result) {
  GameResultTensor outcome = GameResults::win(!active_seat_);
  if (params().announce_game_results) {
    LOG_INFO("Short-circuiting game {} because player {} (seat={}) resigned", game_id_,
             players_[active_seat_]->get_name(), active_seat_);
  }
  handle_terminal(outcome, result);
}

template <concepts::Game Game>
void GameServer<Game>::GameSlot::apply_action(action_t action) {
  using AdvanceUpdate = GameStateTree<Game>::AdvanceUpdate;
  bool is_chance = Rules::is_chance_mode(action_mode_);
  AdvanceUpdate update(state_node_index_, action, active_seat_, is_chance);
  state_node_index_ = state_tree_.advance(update);
}

}  // namespace core
