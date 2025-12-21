#include "core/GameServerProxy.hpp"

#include "core/Constants.hpp"
#include "core/Packet.hpp"
#include "util/Exceptions.hpp"
#include "util/KeyValueDumper.hpp"
#include "util/LoggingUtil.hpp"

#include <arpa/inet.h>
#include <magic_enum/magic_enum.hpp>
#include <magic_enum/magic_enum_format.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>

#include <cstdlib>
#include <cstring>
#include <netdb.h>
#include <string>
#include <unistd.h>

namespace core {

template <concepts::Game Game>
auto GameServerProxy<Game>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("Remote GameServer options");

  return desc
    .template add_option<"remote-server">(
      po::value<std::string>(&remote_server)->default_value(remote_server),
      "Remote server to connect players to")
    .template add_option<"remote-port">(
      po::value<int>(&remote_port),
      "Remote port to connect players to. If not specified, run server in-process");
}

template <concepts::Game Game>
GameServerProxy<Game>::GameSlot::GameSlot(SharedData& shared_data, game_slot_index_t id)
    : shared_data_(shared_data), id_(id) {
  for (player_id_t p = 0; p < (player_id_t)kNumPlayers; ++p) {
    players_[p] = nullptr;
    PlayerGenerator* gen = shared_data_.get_gen(p);
    if (gen) {
      players_[p] = gen->generate(id);
    }
  }
}

template <concepts::Game Game>
GameServerProxy<Game>::GameSlot::~GameSlot() {
  for (player_id_t p = 0; p < (player_id_t)kNumPlayers; ++p) {
    delete players_[p];
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameSlot::handle_start_game(const StartGame& payload) {
  LOG_DEBUG("{}() game_slot={} player_id={}", __func__, payload.game_slot_index, payload.player_id);

  game_id_ = payload.game_id;
  payload.parse_player_names(player_names_);

  game_started_ = true;
  state_tree_.init();
  state_node_index_ = 0;
  mid_yield_ = false;

  Player* player = players_[payload.player_id];
  player->init_game(game_id_, player_names_, payload.seat_assignment);
  player->start_game();
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameSlot::handle_state_change(const StateChange& payload) {
  LOG_DEBUG("{}() id={} game_id={} player_id={}", __func__, id_, game_id_, payload.player_id);

  const char* buf = payload.dynamic_size_section.buf;

  seat_index_t seat = Rules::get_current_player(state());
  ActionResponse action_response;
  std::memcpy(&action_response, buf, sizeof(ActionResponse));
  action_t action = action_response.get_action();
  apply_action(action);

  Player* player = players_[payload.player_id];

  action_mode_t action_mode = Rules::get_action_mode(state());
  StateChangeUpdate update(seat, state(), action, state_node_index_, action_mode);
  player->receive_state_change(update);
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameSlot::handle_action_prompt(const ActionPrompt& payload) {
  LOG_DEBUG("{}() game_slot={} player_id={}", __func__, payload.game_slot_index, payload.player_id);

  const char* buf = payload.dynamic_size_section.buf;
  std::memcpy(&valid_actions_, buf, sizeof(ActionMask));
  prompted_player_id_ = payload.player_id;
  play_noisily_ = payload.play_noisily;

  SlotContext slot_context(id_, 0);
  EnqueueRequest request;
  shared_data_.enqueue(slot_context, request);
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameSlot::handle_end_game(const EndGame& payload) {
  LOG_DEBUG("{}() id={} game_id={} player_id={}", __func__, id_, game_id_, payload.player_id);
  const char* buf = payload.dynamic_size_section.buf;

  game_started_ = false;

  alignas(GameResultTensor) char buffer[sizeof(GameResultTensor)];
  std::memcpy(buffer, buf, sizeof(GameResultTensor));
  GameResultTensor* outcome = new (buffer) GameResultTensor();  // Placement new

  Player* player = players_[payload.player_id];
  player->end_game(state(), *outcome);
}

template <concepts::Game Game>
GameServerBase::StepResult GameServerProxy<Game>::GameSlot::step(context_id_t context) {
  StepResult result;
  EnqueueRequest& enqueue_request = result.enqueue_request;
  Player* player = players_[prompted_player_id_];

  LOG_DEBUG("{}() id={} game_id={} context={} player_id={}", __func__, id_, game_id_, context,
            prompted_player_id_);

  core::action_mode_t mode = Rules::get_action_mode(state());

  // If below assert gets hit, that means we need to add chance-mode support to GameServerProxy.
  // Should be similar to how it works in GameServer.
  //
  // As of yet, we don't even forward chance-event handling prompts from GameServer to
  // GameServerProxy, so it shouldn't be possible to hit this assert.
  RELEASE_ASSERT(!Rules::is_chance_mode(mode), "Unexpected mode: {}", mode);

  YieldNotificationUnit notification_unit(shared_data_.yield_manager(), id_, context);
  ActionRequest request(state(), valid_actions_, notification_unit, get_player_aux());
  request.play_noisily = play_noisily_;

  ActionResponse response = player->get_action_response(request);
  DEBUG_ASSERT(response.extra_enqueue_count == 0 || response.get_yield_instruction() == kYield,
               "Invalid response: extra={} instr={}", response.extra_enqueue_count,
               int(response.get_yield_instruction()));

  switch (response.get_yield_instruction()) {
    case kContinue: {
      CriticalSectionCheck check(in_critical_section_);
      mid_yield_ = false;
      continue_hit_ = true;
      enqueue_request.instruction = kEnqueueLater;
      break;
    }
    case kYield: {
      RELEASE_ASSERT(!continue_hit_, "kYield after continue hit!");
      mid_yield_ = true;
      enqueue_request.instruction = kEnqueueLater;
      enqueue_request.extra_enqueue_count = response.extra_enqueue_count;
      return result;
    }
    case kDrop: {
      enqueue_request.instruction = kEnqueueNever;
      return result;
    }
    default: {
      throw util::Exception("Unexpected response: {}", int(response.get_yield_instruction()));
    }
  }

  if (response.is_aux_set()) {
    set_player_aux(response.aux());
  }

  CriticalSectionCheck check2(in_critical_section_);
  continue_hit_ = false;
  send_action_packet(response);
  return result;
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameSlot::send_action_packet(const ActionResponse& response) {
  Packet<ActionDecision> packet;
  ActionDecision& decision = packet.payload();
  auto& section = decision.dynamic_size_section;

  LOG_DEBUG("{}() id={} game_id={} player_id={} action={}", __func__, id_, game_id_,
            prompted_player_id_, response.get_action());

  decision.game_slot_index = id_;
  decision.player_id = prompted_player_id_;
  packet.set_dynamic_section_size(sizeof(response));
  memcpy(section.buf, &response, sizeof(response));
  packet.send_to(shared_data_.socket());
}

template <concepts::Game Game>
GameServerProxy<Game>::SharedData::SharedData(GameServerProxy* server, const Params& params,
                                              int num_game_threads)
    : server_(server),
      params_(params),
      yield_manager_(cv_, mutex_, queue_, dummy_pending_queue_count_) {
  CLEAN_ASSERT(params_.remote_port > 0, "Remote port must be specified");
  socket_ = io::Socket::create_client_socket(params_.remote_server, params_.remote_port);
  LOG_INFO("Connected to the server!");
}

template <concepts::Game Game>
GameServerProxy<Game>::SharedData::~SharedData() {
  for (auto gen : players_) {
    delete gen;
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::register_player(seat_index_t seat, PlayerGenerator* gen) {
  // TODO: assert that we are not constructing MCTS-T players, since the MCTS-T implementation
  // implicitly assumes that all MCTS-T agents are running in the same process.
  std::string name = gen->get_name();
  CLEAN_ASSERT(name.size() + 1 < kMaxNameLength, "Player name too long (\"{}\" size={})", name,
               name.size());
  seat_generators_.emplace_back(seat, gen);
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::init_socket() {
  int n = seat_generators_.size();
  for (int i = 0; i < n; ++i) {
    SeatGenerator& seat_generator = seat_generators_[i];
    seat_index_t seat = seat_generator.seat;
    PlayerGenerator* gen = seat_generator.gen;

    std::string registered_name = gen->get_name();
    if (registered_name.empty()) {
      registered_name = gen->get_default_name();
    }

    LOG_INFO("Registering player \"{}\" at seat {}", registered_name, seat);

    Packet<Registration> send_packet;
    Registration& registration = send_packet.payload();
    registration.remaining_requests = n - i - 1;
    registration.max_simultaneous_games = gen->max_simultaneous_games();
    registration.requested_seat = seat;
    send_packet.set_player_name(registered_name);
    send_packet.send_to(socket_);

    Packet<RegistrationResponse> recv_packet;
    if (!recv_packet.read_from(socket_)) {
      throw util::Exception("Unexpected socket close");
    }
    const RegistrationResponse& response = recv_packet.payload();
    player_id_t player_id = response.player_id;
    std::string name = response.dynamic_size_section.player_name;

    CLEAN_ASSERT(player_id >= 0 && player_id < kNumPlayers, "Invalid player_id: {}", player_id);
    CLEAN_ASSERT(registered_name.empty() || registered_name == name,
                 "Unexpected name in response: \"{}\" != \"{}\"", registered_name, name);

    gen->set_name(name);
    players_[player_id] = gen;
    LOG_INFO("Registered player \"{}\" at seat {} (player_id:{})", name, seat, player_id);
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::start_session() {
  LOG_INFO("Starting game session");
  for (auto& sg : seat_generators_) {
    sg.gen->start_session();
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::end_session() {
  for (auto& sg : seat_generators_) {
    sg.gen->end_session();
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::shutdown() {
  mit::unique_lock lock(mutex_);
  running_ = false;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::init_game_slots() {
  Packet<GameThreadInitialization> recv_packet;
  if (!recv_packet.read_from(socket_)) {
    throw util::Exception("Unexpected socket close");
  }
  int num_game_slots = recv_packet.payload().num_game_slots;
  game_slots_.reserve(num_game_slots);
  for (int i = 0; i < num_game_slots; ++i) {
    game_slots_.push_back(new GameSlot(*this, i));
  }

  Packet<GameThreadInitializationResponse> send_packet;
  send_packet.send_to(socket_);
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::debug_dump() const {
  mit::unique_lock lock(mutex_);
  LOG_WARN(
    "GameServerProxy {} running:{} queue.size():{} waiting_in_next:{} num_games_started:{} "
    "num_games_ended:{}",
    __func__, running_, queue_.size(), waiting_in_next_, num_games_started_, num_games_ended_);

  for (int i = 0; i < (int)game_slots_.size(); ++i) {
    GameSlot* slot = game_slots_[i];
    bool mid_yield = slot->mid_yield();
    bool continue_hit = slot->continue_hit();
    bool in_critical_section = slot->in_critical_section();

    if (mid_yield || continue_hit || in_critical_section) {
      std::ostringstream ss;
      Game::IO::print_state(ss, slot->state());

      Player* player = slot->prompted_player();
      LOG_WARN(
        "GameServerProxy {} game_slot[{}] mid_yield:{} continue_hit:{} in_critical_section:{} "
        "prompted_player_id:{} prompted_player:{} state:\n{}",
        __func__, i, mid_yield, continue_hit, in_critical_section, slot->prompted_player_id(),
        player ? player->get_name() : "-", ss.str());
    }
  }
}

template <concepts::Game Game>
GameServerBase::next_result_t GameServerProxy<Game>::SharedData::next(SlotContext& item) {
  mit::unique_lock lock(mutex_);

  if (queue_.empty()) {
    LOG_DEBUG("<-- GameServerProxy::{}(): waiting (queue:{})", __func__, queue_.size());
    // NOTE: we need to unlock the mutex before calling force_progress() to avoid a deadlock
    // within NNEvaluationService
    lock.unlock();
    server_->force_progress();
    lock.lock();
    waiting_in_next_ = true;
    cv_.wait(lock, [&] { return !running_ || !queue_pending(); });
    waiting_in_next_ = false;
  }

  if (!running_) {
    LOG_DEBUG("<-- GameServerProxy::{}(): queue empty, exiting", __func__);
    return kExit;
  }

  item = queue_.front();
  queue_.pop();
  LOG_DEBUG("<-- GameServerProxy::{}(): item={}:{} (queue:{})", __func__, item.slot, item.context,
            queue_.size());
  return kProceed;
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::enqueue(SlotContext item, const EnqueueRequest& request) {
  mit::unique_lock lock(mutex_);
  if (request.instruction == kEnqueueNow) {
    RELEASE_ASSERT(request.extra_enqueue_count == 0);
    item.context = 0;  // when continuing, we always want to reset the context to 0
    queue_.push(item);
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
    // pass
  } else {
    throw util::Exception("GameServer::{}(): invalid enqueue instruction: {}", __func__,
                          request.instruction);
  }

  LOG_DEBUG("<-- GameServerProxy::{}(item={}:{}, request={}:{}) queue={}", __func__, item.slot,
            item.context, request.instruction, request.extra_enqueue_count, queue_.size());

  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::handle_start_game(const GeneralPacket& packet) {
  num_games_started_++;
  const StartGame& payload = packet.payload_as<StartGame>();
  game_slots_[payload.game_slot_index]->handle_start_game(payload);
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::handle_state_change(const GeneralPacket& packet) {
  const StateChange& payload = packet.payload_as<StateChange>();
  game_slots_[payload.game_slot_index]->handle_state_change(payload);
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::handle_action_prompt(const GeneralPacket& packet) {
  const ActionPrompt& payload = packet.payload_as<ActionPrompt>();
  game_slots_[payload.game_slot_index]->handle_action_prompt(payload);
}

template <concepts::Game Game>
void GameServerProxy<Game>::SharedData::handle_end_game(const GeneralPacket& packet) {
  num_games_ended_++;
  const EndGame& payload = packet.payload_as<EndGame>();
  game_slots_[payload.game_slot_index]->handle_end_game(payload);
}

template <concepts::Game Game>
GameServerProxy<Game>::GameThread::GameThread(SharedData& shared_data, game_thread_id_t id)
    : shared_data_(shared_data), id_(id) {}

template <concepts::Game Game>
GameServerProxy<Game>::GameThread::~GameThread() {
  join();
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameThread::join() {
  if (thread_.joinable()) thread_.join();
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameThread::launch() {
  thread_ = mit::thread([&] { run(); });
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameThread::run() {
  while (shared_data_.running()) {
    SlotContext item;
    next_result_t next_result = shared_data_.next(item);
    if (next_result == kExit) {
      break;
    } else if (next_result == kProceed) {
      // do nothing
    } else {
      throw util::Exception("Invalid next_result: {}", next_result);
    }

    GameSlot* slot = shared_data_.get_game_slot(item.slot);

    RELEASE_ASSERT(slot->game_started());
    StepResult result = slot->step(item.context);
    shared_data_.enqueue(item, result.enqueue_request);
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::run() {
  Game::static_init();
  shared_data_.init_socket();
  shared_data_.init_game_slots();
  shared_data_.start_session();
  create_threads();
  launch_threads();
  run_event_loop();
  shared_data_.shutdown();
  join_threads();
  shared_data_.end_session();
  util::KeyValueDumper::flush();
}

template <concepts::Game Game>
void GameServerProxy<Game>::create_threads() {
  LOG_INFO("Creating {} game threads", this->num_game_threads());
  for (int t = 0; t < this->num_game_threads(); ++t) {
    GameThread* thread = new GameThread(shared_data_, t);
    threads_.push_back(thread);
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::launch_threads() {
  LOG_INFO("Launching {} game threads", this->num_game_threads());
  for (auto thread : threads_) {
    thread->launch();
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::run_event_loop() {
  LOG_INFO("Entering {}", __func__);
  while (true) {
    GeneralPacket response_packet;
    if (!response_packet.read_from(shared_data_.socket())) {
      break;
    }

    auto type = response_packet.header().type;
    switch (type) {
      case PacketHeader::kStartGame:
        shared_data_.handle_start_game(response_packet);
        break;
      case PacketHeader::kStateChange:
        shared_data_.handle_state_change(response_packet);
        break;
      case PacketHeader::kActionPrompt:
        shared_data_.handle_action_prompt(response_packet);
        break;
      case PacketHeader::kEndGame:
        shared_data_.handle_end_game(response_packet);
        break;
      default:
        throw util::Exception("Unexpected packet type: {}", type);
    }
  }
  LOG_INFO("Exiting {}", __func__);
}

template <concepts::Game Game>
void GameServerProxy<Game>::shutdown_threads() {
  for (auto thread : threads_) {
    thread->shutdown();
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::join_threads() {
  for (auto thread : threads_) {
    thread->join();
  }
}

template <concepts::Game Game>
void GameServerProxy<Game>::GameSlot::apply_action(action_t action) {
  AdvanceUpdate update(state_node_index_, action, prompted_player_id_, false);
  state_node_index_ = state_tree_.advance(update);
}

}  // namespace core
