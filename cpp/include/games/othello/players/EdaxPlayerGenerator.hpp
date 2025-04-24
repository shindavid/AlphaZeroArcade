#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/AbstractPlayerGenerator.hpp>
#include <core/BasicTypes.hpp>
#include <core/OraclePool.hpp>
#include <games/othello/EdaxOracle.hpp>
#include <games/othello/Game.hpp>
#include <games/othello/players/EdaxPlayer.hpp>
#include <util/BoostUtil.hpp>

#include <string>
#include <vector>

namespace othello {

class EdaxPlayerGenerator : public core::AbstractPlayerGenerator<othello::Game> {
 public:
  using Player = core::AbstractPlayer<othello::Game>;

  std::string get_default_name() const override;
  std::vector<std::string> get_types() const override { return {"edax"}; }
  std::string get_description() const override { return "edax player"; }
  Player* generate(core::game_slot_index_t) override;
  void start_session() override;
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;

 private:
  using OraclePool = core::OraclePool<EdaxOracle>;
  EdaxPlayer::Params params_;
  OraclePool oracle_pool_;
};

}  // namespace othello

#include <inline/games/othello/players/EdaxPlayerGenerator.inl>
