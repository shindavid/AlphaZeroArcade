#include <core/TrainingDataWriter.hpp>

#include <filesystem>
#include <string>

#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>
#include <util/StringUtil.hpp>
#include <util/TorchUtil.hpp>

namespace core {

template <concepts::Game Game>
auto TrainingDataWriter<Game>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("TrainingDataWriter options");
  return desc
      .template add_option<"max-rows", 'M'>(
          po::value<int64_t>(&max_rows)->default_value(max_rows),
          "if specified, kill process after writing this many rows")
      .template add_flag<"enable-training", "disable-training">(
          &enabled, "enable training", "disable training");
}

template <concepts::Game Game>
TrainingDataWriter<Game>::TrainingDataWriter(const Params& params)
    : params_(params) {
  if (LoopControllerClient::initialized()) {
    LoopControllerClient::get()->add_listener(this);
  }
  thread_ = new std::thread([&] { loop(); });
}

template <concepts::Game Game>
TrainingDataWriter<Game>::~TrainingDataWriter() {
  shut_down();
}

template <concepts::Game Game>
TrainingDataWriter<Game>::GameLogWriter_sptr
TrainingDataWriter<Game>::make_game_log(game_id_t game_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  game_queue_.push_back(std::make_shared<GameLogWriter>(game_id, util::ns_since_epoch()));
  return game_queue_.back();
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::shut_down() {
  closed_ = true;
  cv_.notify_one();
  if (thread_->joinable()) thread_->join();
  delete thread_;
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::pause() {
  LOG_INFO << "TrainingDataWriter: pausing";
  std::unique_lock lock(mutex_);
  if (paused_) {
    LOG_INFO << "TrainingDataWriter: handle_pause_receipt (already paused)";
    core::LoopControllerClient::get()->handle_pause_receipt();
    return;
  }
  paused_ = true;
  lock.unlock();
  cv_.notify_one();
  LOG_INFO << "TrainingDataWriter: pause complete!";
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::unpause() {
  LOG_INFO << "TrainingDataWriter: unpausing";
  std::unique_lock lock(mutex_);
  if (!paused_) {
    LOG_INFO << "TrainingDataWriter: handle_unpause_receipt (already unpaused)";
    core::LoopControllerClient::get()->handle_unpause_receipt();
    return;
  }
  paused_ = false;
  lock.unlock();
  cv_.notify_one();
  LOG_INFO << "TrainingDataWriter: unpause complete!";
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::loop() {
  std::vector<GameLogWriter_sptr> queue;
  while (!closed_) {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [&] { return (!game_queue_.empty() && game_queue_.front()->done()) || closed_ || paused_; });

    auto it = game_queue_.begin();
    while (it != game_queue_.end() && (*it)->done()) {
      queue.push_back(*it);
      ++it;
    }
    game_queue_.erase(game_queue_.begin(), it);

    if (paused_) {
      LOG_INFO << "TrainingDataWriter: handle_pause_receipt";
      core::LoopControllerClient::get()->handle_pause_receipt();
      LOG_INFO << "TrainingDataWriter: waiting for unpause";
      cv_.wait(lock, [&] { return !paused_; });
      LOG_INFO << "TrainingDataWriter: handle_unpause_receipt";
      core::LoopControllerClient::get()->handle_unpause_receipt();
    }
    lock.unlock();
    for (GameLogWriter_sptr& data : queue) {
      if (send(data.get())) break;
    }
    queue.clear();
  }
}

template <concepts::Game Game>
bool TrainingDataWriter<Game>::send(const GameLogWriter* log) {
  int64_t start_timestamp = log->start_timestamp();
  int64_t cur_timestamp = util::ns_since_epoch();

  core::LoopControllerClient* client = core::LoopControllerClient::get();
  util::release_assert(client, "TrainingDataWriter: no LoopControllerClient");

  int model_generation = client ? client->cur_generation() : 0;

  std::stringstream ss;
  log->serialize(ss);

  int rows = log->sample_count();
  auto new_rows_written = rows_written_ + rows;
  bool done = params_.max_rows > 0 && new_rows_written >= params_.max_rows;
  bool flush = done || client->ready_for_games_flush(cur_timestamp);

  boost::json::object msg;
  msg["type"] = "game";
  msg["gen"] = model_generation;
  msg["start_timestamp"] = start_timestamp;
  msg["end_timestamp"] = cur_timestamp;
  msg["rows"] = rows;
  msg["flush"] = flush;
  msg["done"] = done;

  // TODO: buffer send data and only send when flush==true, waiting for ack before continuing.
  // Also, consider compressing if bandwidth becomes a concern.
  if (flush) {
    if (client->report_metrics()) {
      msg["metrics"] = client->get_perf_stats().to_json();
    }
    client->set_last_games_flush_ts(cur_timestamp);
  }
  client->send_with_file(msg, ss);

  rows_written_ = new_rows_written;
  if (done) {
    LOG_INFO << "TrainingDataWriter: shutting down after writing " << rows_written_ << " rows";
    closed_ = true;
    return true;
  }
  return false;
}

}  // namespace core
