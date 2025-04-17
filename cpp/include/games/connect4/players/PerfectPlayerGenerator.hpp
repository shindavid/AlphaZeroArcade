#pragma once

#include <string>
#include <vector>

#include <core/AbstractPlayer.hpp>
#include <core/AbstractPlayerGenerator.hpp>
#include <core/BasicTypes.hpp>

#include <games/connect4/PerfectOracle.hpp>
#include <games/connect4/players/PerfectPlayer.hpp>

namespace c4 {

class PerfectPlayerGenerator : public core::AbstractPlayerGenerator<c4::Game> {
 public:
  std::string get_default_name() const override;
  std::vector<std::string> get_types() const override { return {"Perfect"}; }
  std::string get_description() const override { return "Perfect player"; }
  core::AbstractPlayer<c4::Game>* generate(core::game_thread_id_t) override;
  void start_session(int num_game_threads) override;
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override;

 private:
  PerfectPlayer::Params params_;
  PerfectOraclePool oracle_pool_;
};

}  // namespace c4

#include <inline/games/connect4/players/PerfectPlayerGenerator.inl>
