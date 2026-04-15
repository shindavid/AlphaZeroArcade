#pragma once

#include "core/AbstractPlayerGenerator.hpp"
#include "core/OraclePool.hpp"
#include "core/PlayerFactory.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/UciProcess.hpp"
#include "games/chess/players/StockfishPlayer.hpp"

namespace a0achess {

class StockfishPlayerGenerator : public core::AbstractPlayerGenerator<Game> {
 public:
  using UciPool = core::OraclePool<UciProcess>;
  using Player = core::AbstractPlayer<Game>;

  StockfishPlayerGenerator(UciPool& pool);

  std::string get_default_name() const override { return std::format("Stockfish-{}", params_.depth); }
  std::string type_str() const override { return "stockfish"; }
  std::string get_description() const override { return "stockfish player"; }
  Player* generate(core::game_slot_index_t) override;
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;


 private:
  UciPlayer::Params params_ = StockfishPlayer::default_params();
  UciPool& pool_;
};

}  // namespace a0achess

namespace core {

template<>
class PlayerSubfactory<a0achess::StockfishPlayerGenerator> : public PlayerSubfactoryBase<a0achess::Game> {
 public:
  using UciPool = a0achess::StockfishPlayerGenerator::UciPool;

  a0achess::StockfishPlayerGenerator* create(GameServerBase*) override {
    return new a0achess::StockfishPlayerGenerator(pool_);
  }

  private:
   UciPool pool_;
};

}  // namespace core

#include "inline/games/chess/players/StockfishPlayerGenerator.inl"
