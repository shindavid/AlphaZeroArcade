#pragma once

#include <core/GameLog.hpp>
#include <core/concepts/Game.hpp>
#include <core/LoopControllerClient.hpp>
#include <core/LoopControllerListener.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <cstdint>
#include <deque>
#include <list>
#include <mutex>
#include <thread>

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

  using GameLogWriter = core::GameLogWriter<Game>;
  using GameLogWriter_sptr = std::shared_ptr<GameLogWriter>;

  TrainingDataWriter(const Params& params);
  ~TrainingDataWriter();

  GameLogWriter_sptr make_game_log(game_id_t game_id);
  void notify() {cv_.notify_one();}
  void shut_down();

  void pause() override;
  void unpause() override;

 protected:
  using game_queue_t = std::deque<GameLogWriter_sptr>;

  void loop();
  bool send(const GameLogWriter* log);  // return true if this is last game

  Params params_;
  std::thread* thread_;
  game_queue_t game_queue_;
  int64_t rows_written_ = 0;

  bool closed_ = false;
  bool paused_ = false;

  std::condition_variable cv_;
  mutable std::mutex mutex_;
};

}  // namespace core

#include <inline/core/TrainingDataWriter.inl>
