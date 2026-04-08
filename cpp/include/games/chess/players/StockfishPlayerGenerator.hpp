#pragma once

#include "core/AbstractPlayerGenerator.hpp"
#include "core/OraclePool.hpp"
#include "core/PlayerFactory.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/StockfishProcess.hpp"
#include "games/chess/players/StockfishPlayer.hpp"

namespace a0achess {

class StockfishPlayerGenerator : public core::AbstractPlayerGenerator<Game> {
 public:
  using StockfishPool = core::OraclePool<StockfishProcess>;
  using Player = core::AbstractPlayer<Game>;

  StockfishPlayerGenerator(StockfishPool& stockfish_pool);

  std::string get_default_name() const override { return std::format("Stockfish-{}", params_.depth); }
  std::string type_str() const override { return "stockfish"; }
  std::string get_description() const override { return "stockfish player"; }
  Player* generate(core::game_slot_index_t) override;
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;


 private:
  StockfishPlayer::Params params_;
  StockfishPool& stockfish_pool_;
};

}  // namespace a0achess

namespace core {

template<>
class PlayerSubfactory<a0achess::StockfishPlayerGenerator> : public PlayerSubfactoryBase<a0achess::Game> {
 public:
  using StockfishPool = a0achess::StockfishPlayerGenerator::StockfishPool;

  a0achess::StockfishPlayerGenerator* create(GameServerBase*) override {
    return new a0achess::StockfishPlayerGenerator(stockfish_pool_);
  }

  private:
   StockfishPool stockfish_pool_;
};

}  // namespace core

#include "inline/games/chess/players/StockfishPlayerGenerator.inl"
