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
    bool enabled = false;
  };

  using ValueArray = Game::Types::ValueArray;
  using InputTensorizor = Game::InputTensorizor;
  using TrainingTargetsList = Game::TrainingTargets::List;

  using GameWriteLog = core::GameWriteLog<Game>;
  using GameWriteLog_sptr = std::shared_ptr<GameWriteLog>;

  TrainingDataWriter(const Params& params);
  ~TrainingDataWriter();

  void add(GameWriteLog_sptr log);
  void shut_down();

  void pause() override;
  void unpause() override;

 protected:
  using game_queue_t = std::vector<GameWriteLog_sptr>;

  void loop();
  bool send(const GameWriteLog* log);  // return true if this is last game

  Params params_;
  std::thread* thread_;
  game_queue_t completed_games_[2];
  int64_t rows_written_ = 0;

  int queue_index_ = 0;
  bool closed_ = false;
  bool paused_ = false;
  bool direct_game_log_write_optimization_enabled_ = false;

  int last_created_dir_generation_ = -1;  // for direct-game-log-write optimization

  std::condition_variable cv_;
  mutable std::mutex mutex_;
};

}  // namespace core

#include <inline/core/TrainingDataWriter.inl>
