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
    LoopControllerClient* client = LoopControllerClient::get();
    client->add_listener(this);
    if (client->is_loop_controller_local()) {
      if (client->output_base_dir().empty()) {
        LOG_WARN << "--output-base-dir not set despite using a local loop controller";
        LOG_WARN << "Disabling direct-game-log-write optimization";
      } else {
        direct_game_log_write_optimization_enabled_ = true;
      }
    }
  }
  thread_ = new std::thread([&] { loop(); });
}

template <concepts::Game Game>
TrainingDataWriter<Game>::~TrainingDataWriter() {
  shut_down();
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::add(GameLogWriter_sptr data) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_games_[queue_index_].push_back(data);
  lock.unlock();
  cv_.notify_one();
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
  while (!closed_) {
    std::unique_lock lock(mutex_);
    game_queue_t& queue = completed_games_[queue_index_];
    cv_.wait(lock, [&] { return !queue.empty() || closed_ || paused_; });
    if (paused_) {
      LOG_INFO << "TrainingDataWriter: handle_pause_receipt";
      core::LoopControllerClient::get()->handle_pause_receipt();
      LOG_INFO << "TrainingDataWriter: waiting for unpause";
      cv_.wait(lock, [&] { return !paused_; });
      LOG_INFO << "TrainingDataWriter: handle_unpause_receipt";
      core::LoopControllerClient::get()->handle_unpause_receipt();
    }
    queue_index_ = 1 - queue_index_;
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

  // Optimization: if we are on the same machine as the loop-controller, we can write the file
  // directly to the filesystem, rather than sending it over the TCP socket. This helps reduce
  // socket contention, which empirically can be a bottleneck in some settings.
  //
  // Some technical notes:
  //
  // 1. The loop-controller's SelfPlayManager has the responsibility to both write the file to disk
  //    AND to add sqlite database entries. It is important to do so in that order, in case there is
  //    a crash in between the two operations (a file on disk without a corresponding database entry
  //    is essentially just ignored, but a database entry without a file will lead to a crash).
  //
  // 2. The loop-controller's SelfPlayManager MUST be the one to do the database update, because
  //    sqlite3 only permits a single writer to a database at a time. There are some ways around
  //    this, but in my experience they are not worth the trouble.
  //
  // 3. Therefore, for this optimization, we must first write the file to disk here, and then send a
  //    message to the loop-controller to update the database, in that order.
  //
  // 4. The loop-controller's SelfPlayManager will sometimes intentionally drop certain games,
  //    skipping the filesystem-write and the database-update. Unless we copy that logic from
  //    python into c++ (which we don't want to for code-maintenance reasons), we lose the ability
  //    to replicate that behavior. That is ok, because again, the filesystem-write without a
  //    database-update is essentially a drop.
  if (direct_game_log_write_optimization_enabled_) {

    // In the future, if we change the logic controlling the game filename on the python-side, we
    // need to change this code to match the python-side. Not ideal, but it is what it is.
    std::string directory =
        util::create_string("%s/gens/gen-%d/self-play/client-%d", client->output_base_dir().c_str(),
                            model_generation, client->client_id());

    if (model_generation != last_created_dir_generation_) {
      boost::filesystem::create_directories(directory);  // mkdir -p
      last_created_dir_generation_ = model_generation;
    }

    std::string filename = util::create_string("%s/%ld.log", directory.c_str(), cur_timestamp);

    std::ofstream file(filename);
    file << ss.str();
    file.close();

    msg["no-file"] = true;
    client->send(msg);
  } else {
    client->send_with_file(msg, ss);
  }

  rows_written_ = new_rows_written;
  if (done) {
    LOG_INFO << "TrainingDataWriter: shutting down after writing " << rows_written_ << " rows";
    closed_ = true;
    return true;
  }
  return false;
}

}  // namespace core
