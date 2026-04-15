#pragma once

#include "core/AbstractPlayerGenerator.hpp"
#include "core/OraclePool.hpp"
#include "core/PlayerFactory.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/LcZeroProcess.hpp"
#include "games/chess/players/LcZeroPlayer.hpp"

namespace a0achess {

class LcZeroPlayerGenerator : public core::AbstractPlayerGenerator<Game> {
 public:
  using LcZeroPool = core::OraclePool<LcZeroProcess>;
  using Player = core::AbstractPlayer<Game>;

  LcZeroPlayerGenerator(LcZeroPool& lc0_pool);

  std::string get_default_name() const override { return std::format("lc0-{}", params_.movetime); }
  std::string type_str() const override { return "lc0"; }
  std::string get_description() const override { return "lc0 player"; }
  Player* generate(core::game_slot_index_t) override;
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;


 private:
  LcZeroPlayer::Params params_;
  LcZeroPool& lc0_pool_;
};

}  // namespace a0achess

namespace core {

template<>
class PlayerSubfactory<a0achess::LcZeroPlayerGenerator> : public PlayerSubfactoryBase<a0achess::Game> {
 public:
  using LcZeroPool = a0achess::LcZeroPlayerGenerator::LcZeroPool;

  a0achess::LcZeroPlayerGenerator* create(GameServerBase*) override {
    return new a0achess::LcZeroPlayerGenerator(lc0_pool_);
  }

  private:
   LcZeroPool lc0_pool_;
};

}  // namespace core

#include "inline/games/chess/players/LcZeroPlayerGenerator.inl"
