#pragma once

#include <cstdint>
#include <vector>

namespace core {

class GameServerClient;

class GameServerBase {
 public:
  enum enqueue_instruction_t : int8_t { kEnqueueNow, kEnqueueLater, kEnqueueNever };

  virtual ~GameServerBase() = default;

  struct EnqueueRequest {
    enqueue_instruction_t instruction = kEnqueueNow;
    int extra_enqueue_count = 0;  // used when instruction == kEnqueueLater
  };

  GameServerBase(int num_game_threads) : num_game_threads_(num_game_threads) {
    game_servers_.push_back(this);
  }

  static void add_client(GameServerClient* client);

  int num_game_threads() const {
    return num_game_threads_;
  }

  virtual void debug_dump() const = 0;

 protected:
  void force_progress();

 private:
  using server_vec_t = std::vector<GameServerBase*>;
  using client_vec_t = std::vector<GameServerClient*>;

  static server_vec_t game_servers_;
  client_vec_t clients_;
  const int num_game_threads_;
};

}  // namespace core
