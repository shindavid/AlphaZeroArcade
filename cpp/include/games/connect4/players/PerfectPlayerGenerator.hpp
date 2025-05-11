#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/AbstractPlayerGenerator.hpp>
#include <core/BasicTypes.hpp>
#include <core/GameServerBase.hpp>
#include <core/OraclePool.hpp>
#include <core/PlayerFactory.hpp>
#include <games/connect4/PerfectOracle.hpp>
#include <games/connect4/players/PerfectPlayer.hpp>

#include <string>
#include <vector>

namespace c4 {

class PerfectPlayerGenerator : public core::AbstractPlayerGenerator<c4::Game> {
 public:
  using OraclePool = core::OraclePool<PerfectOracle>;
  using Player = core::AbstractPlayer<c4::Game>;

  PerfectPlayerGenerator(core::GameServerBase* server, OraclePool& oracle_pool)
      : server_(server), oracle_pool_(oracle_pool) {}

  std::string get_default_name() const override;
  std::vector<std::string> get_types() const override { return {"Perfect"}; }
  std::string get_description() const override { return "Perfect player"; }
  Player* generate(core::game_slot_index_t) override;
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;
  void start_session() override;

 private:
  PerfectPlayer::Params params_;
  core::GameServerBase* server_;
  OraclePool& oracle_pool_;
};

}  // namespace c4

namespace core {

template <>
class PlayerSubfactory<c4::PerfectPlayerGenerator> : public PlayerSubfactoryBase<c4::Game> {
 public:
  using OraclePool = c4::PerfectPlayerGenerator::OraclePool;

  c4::PerfectPlayerGenerator* create(GameServerBase* server) override {
    return new c4::PerfectPlayerGenerator(server, oracle_pool_);
  }

 private:
  OraclePool oracle_pool_;
};

}  // namespace core

#include <inline/games/connect4/players/PerfectPlayerGenerator.inl>
