#pragma once

#include "core/AbstractPlayerGenerator.hpp"
#include "core/OraclePool.hpp"
#include "core/PlayerFactory.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/UciProcess.hpp"
#include "games/chess/players/LcZeroPlayer.hpp"

namespace a0achess {

class LcZeroPlayerGenerator : public core::AbstractPlayerGenerator<Game> {
 public:
  using UciPool = core::OraclePool<UciProcess>;
  using Player = core::AbstractPlayer<Game>;

  LcZeroPlayerGenerator(UciPool& pool);

  std::string get_default_name() const override { return std::format("lc0-{}", params_.movetime); }
  std::string type_str() const override { return "lc0"; }
  std::string get_description() const override { return "lc0 player"; }
  Player* generate(core::game_slot_index_t) override;
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;


 private:
  UciPlayer::Params params_ = LcZeroPlayer::default_params();
  UciPool& pool_;
};

}  // namespace a0achess

namespace core {

template<>
class PlayerSubfactory<a0achess::LcZeroPlayerGenerator> : public PlayerSubfactoryBase<a0achess::Game> {
 public:
  using UciPool = a0achess::LcZeroPlayerGenerator::UciPool;

  a0achess::LcZeroPlayerGenerator* create(GameServerBase*) override {
    return new a0achess::LcZeroPlayerGenerator(pool_);
  }

  private:
   UciPool pool_;
};

}  // namespace core

#include "inline/games/chess/players/LcZeroPlayerGenerator.inl"
