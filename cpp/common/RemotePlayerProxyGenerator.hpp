#pragma once

#include <string>
#include <vector>

#include <common/AbstractPlayerGenerator.hpp>
#include <common/BasicTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/RemotePlayerProxy.hpp>

namespace common {

template <GameStateConcept GameState>
class RemotePlayerProxyGenerator : public AbstractPlayerGenerator<GameState> {
 public:
  using base_t = AbstractPlayerGenerator<GameState>;

  void initialize(const std::string& name, int socket_descriptor, player_id_t player_id);
  bool initialized() const { return socket_descriptor_ != -1; }
  int get_socket_descriptor() const { return socket_descriptor_; }

  std::vector<std::string> get_types() const override { return { "Remote" }; }
  std::string get_description() const override { return "Remote player from another process"; }
  AbstractPlayer<GameState>* generate(game_thread_id_t) override;
  void end_session() override;

 private:
  int socket_descriptor_ = -1;
  player_id_t player_id_ = -1;
};

}  // namespace common

#include <common/inl/RemotePlayerProxyGenerator.inl>
