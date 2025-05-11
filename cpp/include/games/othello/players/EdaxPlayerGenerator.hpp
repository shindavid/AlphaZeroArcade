#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/AbstractPlayerGenerator.hpp>
#include <core/BasicTypes.hpp>
#include <core/GameServerBase.hpp>
#include <core/OraclePool.hpp>
#include <core/PlayerFactory.hpp>
#include <games/othello/EdaxOracle.hpp>
#include <games/othello/Game.hpp>
#include <games/othello/players/EdaxPlayer.hpp>

#include <string>
#include <vector>

namespace othello {

class EdaxPlayerGenerator : public core::AbstractPlayerGenerator<othello::Game> {
 public:
  using OraclePool = core::OraclePool<EdaxOracle>;
  using Player = core::AbstractPlayer<othello::Game>;

  EdaxPlayerGenerator(core::GameServerBase* server, OraclePool& oracle_pool)
      : server_(server), oracle_pool_(oracle_pool) {}

  std::string get_default_name() const override;
  std::vector<std::string> get_types() const override { return {"edax"}; }
  std::string get_description() const override { return "edax player"; }
  Player* generate(core::game_slot_index_t) override;
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;
  void start_session() override;

 private:
  EdaxPlayer::Params params_;
  core::GameServerBase* server_;
  OraclePool& oracle_pool_;
};

}  // namespace othello

namespace core {

template <>
class PlayerSubfactory<othello::EdaxPlayerGenerator> : public PlayerSubfactoryBase<othello::Game> {
 public:
  using OraclePool = othello::EdaxPlayerGenerator::OraclePool;

  othello::EdaxPlayerGenerator* create(GameServerBase* server) override {
    return new othello::EdaxPlayerGenerator(server, oracle_pool_);
  }

 private:
  OraclePool oracle_pool_;
};

}  // namespace core

#include <inline/games/othello/players/EdaxPlayerGenerator.inl>
