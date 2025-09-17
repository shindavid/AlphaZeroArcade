#pragma once

#include "core/LoopControllerListener.hpp"
#include "core/TrainingParams.hpp"
#include "search/GameLog.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <chrono>
#include <map>
#include <vector>

namespace search {

/*
 * Noteworthy implementation details:
 *
 ****************************
 * Data Member Organization *
 ****************************
 *
 * We organization the key data members into 3 structs: GameQueueData, BatchData, and MiscData.
 * We do this separation to make it clear which members are controlled by which mutex.
 *
 *****************************************
 * GameQueueData Two-Vector Optimization *
 *****************************************
 *
 * GameQueueData's key data member is effectively a queue of completed games. Whenever a game is
 * completed, the associated game log is added to this queue, to be processed by the loop() method.
 * This processing entails serializing the game into sequence of bytes, which get added to a data
 * batch.
 *
 * A naive implementation of this queue might use a single std::vector. This would require us to
 * hold a mutex lock while we are performing the serialization, which would block the game threads
 * as they wait to add to the queue. That isn't ideal. We could consider copying the vector first,
 * and then releasing the lock, and doing the serialization on the copy.
 *
 * However, we can do better. We can use a two-vector optimization. Instead of making a copy, we
 * maintain two vectors, and we switch between them. The game threads add to the inactive vector,
 * while the loop() method processes the active vector. The mutex only needs to protect access to
 * the active vector, and to the bit that tells us which vector is active.
 *
 **********************************
 * Batch Data and Heartbeat Logic *
 **********************************
 *
 * The loop-controller requires a specific number of rows that it needs in order to move onto the
 * next generation. The goal is for the TrainingDataWriter to accumulate game data until it has
 * enough rows to meet this requirement, and then to send that data to the loop-controller in a
 * single batch. Compared to sending each individual game as it is completed, this batching is
 * more efficient, particularly from a filesystem I/O perspective on the loop-controller side.
 *
 * In order to facilitate this, the loop-controller needs to keep track of the amount of data
 * accumulated by the self-play server(s). When that total reaches the required number of rows, the
 * loop-controller can then issue a request for the data.
 *
 * How can it keep track of this number, without direct visibility into the c++ process? This is the
 * purpose of the heartbeat mechanism: the TrainingDataWriter periodically sends a heartbeat message
 * to the loop-controller informing it of how many rows are in its current batch.
 *
 * As a minor optimization, we have the loop-controller tell the TrainingDataWriter at the start of
 * the generation the exact number of rows it needs. This way, when the TrainingDataWriter
 * accumulates that many rows, it can send a heartbeat immediately, rather than waiting for the next
 * heartbeat interval. Note that in the multi-self-play-server case, each of the self-play servers
 * can produce more data in total than the loop controller actually needs. This is fine; the
 * loop-controller will simply ignore the extra data.
 */
template <search::concepts::Traits Traits>
class TrainingDataWriter
    : public core::LoopControllerListener<core::LoopControllerInteractionType::kPause>,
      public core::LoopControllerListener<core::LoopControllerInteractionType::kDataRequest> {
 public:
  using Game = Traits::Game;
  using ValueArray = Game::Types::ValueArray;

  using GameLogSerializer = search::GameLogSerializer<Traits>;
  using GameWriteLog = search::GameWriteLog<Traits>;
  using GameWriteLog_sptr = std::shared_ptr<GameWriteLog>;

  static TrainingDataWriter* instance();
  ~TrainingDataWriter();

  GameWriteLog_sptr get_game_log(core::game_id_t id);
  void add(GameWriteLog_sptr log);
  void shut_down();
  bool closed() const { return misc_data_.closed; }
  void wait_until_batch_empty();

  void pause() override;
  void unpause() override;
  void handle_data_request(int n_rows, int next_row_limit) override;
  void handle_data_pre_request(int n_rows_limit) override;

 protected:
  using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
  using game_queue_t = std::vector<GameWriteLog_sptr>;

  TrainingDataWriter(const core::TrainingParams& params);
  const auto& heartbeat_interval() const { return misc_data_.heartbeat_interval; }

  void loop();
  void record(const GameWriteLog* log);  // return true if batch is full
  void send_batch(int n_rows);
  void send_heartbeat();

  /*
   * Data members used to control access to game log queue.
   *
   * Access to this data must be protected by game_queue_mutex_.
   */
  struct GameQueueData {
    game_queue_t completed_games[2];
    int queue_index = 0;
    bool paused = false;
  };

  /*
   * Data members used to control access to batch data sent to the loop-controller.
   *
   * Access to this data must be protected by batch_mutex_.
   */
  struct BatchData {
    bool full() const { return limit > 0 && size >= limit; }
    void reset();

    time_point_t next_heartbeat_time;
    int limit = 0;
    int size = 0;
    int last_heartbeat_size = 0;
    std::vector<GameLogMetadata> metadata;
    std::vector<char> data;
    std::vector<char> send_buf;
  };

  /*
   * These data members do not require mutex protection, as they are not accessed concurrently.
   */
  struct MiscData {
    core::TrainingParams params;
    mit::thread* thread;
    std::chrono::nanoseconds heartbeat_interval;
    bool closed = false;
    bool direct_game_log_write_optimization_enabled = false;
    int last_created_dir_generation = -1;  // for direct-game-log-write optimization
    int num_game_threads;
  };

  MiscData misc_data_;
  GameQueueData game_queue_data_;
  BatchData batch_data_;
  GameLogSerializer serializer_;
  std::map<core::game_id_t, GameWriteLog_sptr> active_logs_;

  mit::condition_variable game_queue_cv_;
  mit::condition_variable batch_cv_;
  mutable mit::mutex game_queue_mutex_;
  mutable mit::mutex batch_mutex_;
  mutable mit::mutex active_logs_mutex_;
};

}  // namespace search

#include "inline/search/TrainingDataWriter.inl"
