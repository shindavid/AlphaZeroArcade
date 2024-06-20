#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <core/GameLog.hpp>
#include <core/concepts/Game.hpp>
#include <core/LoopControllerClient.hpp>
#include <core/LoopControllerListener.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>

namespace core {

/*
 * A single TrainingDataWriter is intended to be shared by multiple MctsPlayer's playing in
 * parallel.
 */
template <concepts::Game Game>
class TrainingDataWriter
    : public core::LoopControllerListener<core::LoopControllerInteractionType::kPause> {
 public:
  struct Params {
    auto make_options_description();
    bool operator==(const Params& other) const = default;

    int64_t max_rows = 0;
  };

  using ValueArray = typename Game::Types::ValueArray;
  using InputTensorizor = typename Game::InputTensorizor;
  using TrainingTargetTensorizor = typename Game::TrainingTargetTensorizor;

  using GameLogWriter = core::GameLogWriter<Game>;
  using GameLogWriter_sptr = std::shared_ptr<GameLogWriter>;
  using game_log_map_t = std::map<game_id_t, GameLogWriter_sptr>;

  static TrainingDataWriter* instantiate(const Params& params);

  /*
   * Assumes that instantiate() was called at least once.
   */
  static TrainingDataWriter* instance() { return instance_; }

  GameLogWriter_sptr get_log(game_id_t id);

  void close(GameLogWriter_sptr log);
  void shut_down();

  void pause() override;
  void unpause() override;

 protected:
  using game_queue_t = std::vector<GameLogWriter_sptr>;

  TrainingDataWriter(const Params& params);
  ~TrainingDataWriter();

  void loop();
  bool send(const GameLogWriter* log);  // return true if this is last game

  Params params_;
  std::thread* thread_;
  game_log_map_t game_log_map_;
  game_queue_t completed_games_[2];
  int64_t rows_written_ = 0;

  int queue_index_ = 0;
  bool closed_ = false;
  bool paused_ = false;

  std::condition_variable cv_;
  mutable std::mutex mutex_;

  static TrainingDataWriter* instance_;
};

}  // namespace core

#include <inline/core/TrainingDataWriter.inl>
