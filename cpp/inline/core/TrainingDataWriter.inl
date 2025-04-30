#include <core/TrainingDataWriter.hpp>

#include <format>
#include <string>

#include <core/Globals.hpp>
#include <core/PerfStats.hpp>
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
      .template add_option<"heartbeat-frequency-seconds", 'H'>(
          po::value<float>(&heartbeat_frequency_seconds)
              ->default_value(heartbeat_frequency_seconds),
          "heartbeat frequency in seconds")
      .template add_flag<"enable-training", "disable-training">(&enabled, "enable training",
                                                                "disable training");
}

template <concepts::Game Game>
TrainingDataWriter<Game>::TrainingDataWriter(const Params& params) {
  misc_data_.params = params;
  if (LoopControllerClient::initialized()) {
    LoopControllerClient* client = LoopControllerClient::get();
    client->add_listener(this);
    if (client->is_loop_controller_local()) {
      if (client->output_base_dir().empty()) {
        LOG_WARN("--output-base-dir not set despite using a local loop controller");
        LOG_WARN("Disabling direct-game-log-write optimization");
      } else {
        misc_data_.direct_game_log_write_optimization_enabled = true;
      }
    }
  }
  batch_data_.limit = params.max_rows;
  batch_data_.next_heartbeat_time = std::chrono::steady_clock::now();
  misc_data_.heartbeat_interval =
      std::chrono::milliseconds(int64_t(1e3 * params.heartbeat_frequency_seconds));
  misc_data_.thread = new std::thread([&] { loop(); });
}

template <concepts::Game Game>
TrainingDataWriter<Game>::~TrainingDataWriter() {
  shut_down();
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::add(GameWriteLog_sptr data) {
  std::unique_lock lock(game_queue_mutex_);
  game_queue_data_.completed_games[game_queue_data_.queue_index].push_back(data);
  lock.unlock();
  game_queue_cv_.notify_one();
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::shut_down() {
  LOG_INFO("TrainingDataWriter: shutting down");
  misc_data_.closed = true;
  game_queue_cv_.notify_one();
  if (misc_data_.thread->joinable()) misc_data_.thread->join();
  delete misc_data_.thread;
  LOG_INFO("TrainingDataWriter: shutdown complete!");
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::wait_until_batch_empty() {
  std::unique_lock lock(batch_mutex_);
  batch_cv_.wait(lock, [&] { return batch_data_.size == 0; });
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::pause() {
  LOG_INFO("TrainingDataWriter: pausing");
  std::unique_lock lock(game_queue_mutex_);
  if (game_queue_data_.paused) {
    LOG_INFO("TrainingDataWriter: handle_pause_receipt (already paused)");
    core::LoopControllerClient::get()->handle_pause_receipt();
    return;
  }
  game_queue_data_.paused = true;
  lock.unlock();
  game_queue_cv_.notify_one();
  LOG_INFO("TrainingDataWriter: pause complete!");
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::unpause() {
  LOG_INFO("TrainingDataWriter: unpausing");
  std::unique_lock lock(game_queue_mutex_);
  if (!game_queue_data_.paused) {
    LOG_INFO("TrainingDataWriter: handle_unpause_receipt (already unpaused)");
    core::LoopControllerClient::get()->handle_unpause_receipt();
    return;
  }
  game_queue_data_.paused = false;
  lock.unlock();
  game_queue_cv_.notify_one();
  LOG_INFO("TrainingDataWriter: unpause complete!");
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::handle_data_request(int n_rows) {
  std::unique_lock lock(batch_mutex_);

  send_batch(n_rows);
  batch_data_.reset();
  batch_data_.next_heartbeat_time = std::chrono::steady_clock::now() + heartbeat_interval();
  batch_data_.limit = 0;
  batch_cv_.notify_one();
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::handle_data_pre_request(int n_rows_limit) {
  std::unique_lock lock(batch_mutex_);
  batch_data_.limit = n_rows_limit;
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::loop() {
  while (!misc_data_.closed) {
    std::unique_lock lock(game_queue_mutex_);
    game_queue_t& queue = game_queue_data_.completed_games[game_queue_data_.queue_index];
    game_queue_cv_.wait(lock, [&] {
      return !queue.empty() || misc_data_.closed || game_queue_data_.paused;
    });
    if (game_queue_data_.paused) {
      core::LoopControllerClient::get()->handle_pause_receipt();
      game_queue_cv_.wait(lock, [&] { return !game_queue_data_.paused; });
      core::LoopControllerClient::get()->handle_unpause_receipt();
    }
    game_queue_data_.queue_index = 1 - game_queue_data_.queue_index;
    lock.unlock();

    // ok to access queue without a lock from here on

    std::unique_lock batch_lock(batch_mutex_);
    for (GameWriteLog_sptr& data : queue) {
      if (batch_data_.full()) {
        misc_data_.closed |= (misc_data_.params.max_rows > 0);
        break;
      }
      record(data.get());
    }
    queue.clear();

    if (batch_data_.size == batch_data_.last_heartbeat_size) continue;

    auto now = std::chrono::steady_clock::now();
    if (batch_data_.full() || now > batch_data_.next_heartbeat_time) {
      send_heartbeat();
      batch_data_.next_heartbeat_time = now + heartbeat_interval();
    }
  }
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::record(const GameWriteLog* log) {
  // assumes that batch_mutex_ is locked
  auto client = core::LoopControllerClient::get();
  int client_id = client ? client->client_id() : 0;
  batch_data_.metadata.push_back(serializer_.serialize(log, batch_data_.data, client_id));
  batch_data_.size += log->sample_count();
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::send_batch(int n_rows) {
  // assumes that batch_mutex_ is locked

  int n_total_games = batch_data_.metadata.size();

  // First, figure out how many games to truncate in order to respect n_rows
  int row_count = 0;
  int n_games = 0;
  for (; n_games < n_total_games; ++n_games) {
    row_count += batch_data_.metadata[n_games].num_samples;
    if (row_count >= n_rows) {
      n_games++;
      break;
    }
  }

  util::release_assert(n_rows > 0, "TrainingDataWriter: n_rows <= 0 ({})", n_rows);
  util::release_assert(row_count >= n_rows, "TrainingDataWriter: row_count < n_rows ({} < {})",
                       row_count, n_rows);
  util::release_assert(n_games > 0, "TrainingDataWriter: n_games <= 0 ({})", n_games);

  LOG_INFO("TrainingDataWriter: sending batch of {} rows from {} games (of {} total)", row_count,
           n_games, n_total_games);

  // Truncate
  batch_data_.metadata.resize(n_games);
  const auto& last_metadata = batch_data_.metadata.back();
  int truncated_data_size = last_metadata.start_offset + last_metadata.data_size;
  batch_data_.data.resize(truncated_data_size);

  // Recalibrate the start_offset of each GameLogMetadata
  uint32_t offset_adjustment = sizeof(GameLogFileHeader) + sizeof(GameLogMetadata) * n_games;
  for (auto& md : batch_data_.metadata) {
    md.start_offset += offset_adjustment;
  }

  GameLogFileHeader header;
  header.num_games = n_games;
  header.num_rows = row_count;

  batch_data_.send_buf.clear();
  GameLogCommon::write_section(batch_data_.send_buf, &header);
  GameLogCommon::write_section(batch_data_.send_buf, batch_data_.metadata.data(), n_games);
  batch_data_.send_buf.insert(batch_data_.send_buf.end(), batch_data_.data.begin(),
                              batch_data_.data.end());

  core::LoopControllerClient* client = core::LoopControllerClient::get();
  util::release_assert(client, "TrainingDataWriter: no LoopControllerClient");

  int model_generation = client ? client->cur_generation() : 0;

  boost::json::object msg;
  msg["type"] = "self-play-data";
  msg["timestamp"] = util::ns_since_epoch();
  msg["gen"] = model_generation;
  msg["n_games"] = n_games;
  msg["n_rows"] = row_count;
  if (client->report_metrics()) {
    PerfStats stats = core::PerfStatsRegistry::instance()->get_perf_stats();
    stats.calibrate(core::Globals::num_game_threads);
    msg["metrics"] = stats.to_json();
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
  if (misc_data_.direct_game_log_write_optimization_enabled) {
    // In the future, if we change the logic controlling the game filename on the python-side, we
    // need to change this code to match the python-side. Not ideal, but it is what it is.
    std::string filename =
      std::format("/home/devuser/scratch/self-play-data/{}.data", client->client_id());

    std::ofstream file(filename, std::ios::binary);
    file.write(batch_data_.send_buf.data(), batch_data_.send_buf.size());
    util::release_assert(file.good(), "TrainingDataWriter: failed to write to file {}", filename);
    file.close();

    msg["no_file"] = true;
    client->send(msg);
  } else {
    client->send_with_file(msg, batch_data_.send_buf);
  }
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::send_heartbeat() {
  // assumes that batch_mutex_ is locked

  core::LoopControllerClient* client = core::LoopControllerClient::get();

  boost::json::object msg;
  msg["type"] = "heartbeat";
  msg["rows"] = batch_data_.size;
  client->send(msg);

  batch_data_.last_heartbeat_size = batch_data_.size;
}

template <concepts::Game Game>
void TrainingDataWriter<Game>::BatchData::reset() {
  // assumes that batch_mutex_ is locked

  size = 0;
  last_heartbeat_size = 0;
  metadata.clear();
  data.clear();
}

}  // namespace core
