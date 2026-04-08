#include "games/chess/players/StockfishPlayerGenerator.hpp"

namespace a0achess {

inline StockfishPlayerGenerator::StockfishPlayerGenerator(StockfishPool& stockfish_pool)
    : stockfish_pool_(stockfish_pool) {}

inline core::AbstractPlayer<Game>* StockfishPlayerGenerator::generate(core::game_slot_index_t) {
  return new StockfishPlayer(&stockfish_pool_, params_);
}

inline void StockfishPlayerGenerator::parse_args(const std::vector<std::string>& args) {
  namespace po2 = boost_util::program_options;
  po2::parse_args(params_.make_options_description(), args);

  size_t capacity = params_.num_stockfish_procs;
  if (capacity > stockfish_pool_.capacity()) {
    stockfish_pool_.set_capacity(capacity);
  }
}

}  // namespace a0achess
