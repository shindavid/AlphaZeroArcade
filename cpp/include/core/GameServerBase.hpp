#pragma once

#include <cstdint>
#include <vector>

namespace core {

class GameServerClient;

class GameServerBase {
 public:
  enum enqueue_instruction_t : int8_t { kEnqueueNow, kEnqueueLater, kEnqueueNever };

  struct EnqueueRequest {
    enqueue_instruction_t instruction = kEnqueueNow;
    int extra_enqueue_count = 0;  // used when instruction == kEnqueueLater
  };

  GameServerBase() { game_servers_.push_back(this); }
  static void add_client(GameServerClient* client);

 protected:
  void force_progress();

 private:
  using server_vec_t = std::vector<GameServerBase*>;
  using client_vec_t = std::vector<GameServerClient*>;

  static server_vec_t game_servers_;
  client_vec_t clients_;
};

}  // namespace core
