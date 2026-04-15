#include "games/chess/players/LcZeroPlayerGenerator.hpp"

namespace a0achess {

inline LcZeroPlayerGenerator::LcZeroPlayerGenerator(LcZeroPool& lc0_pool)
    : lc0_pool_(lc0_pool) {}

inline core::AbstractPlayer<Game>* LcZeroPlayerGenerator::generate(core::game_slot_index_t) {
  return new LcZeroPlayer(&lc0_pool_, params_);
}

inline void LcZeroPlayerGenerator::parse_args(const std::vector<std::string>& args) {
  namespace po2 = boost_util::program_options;
  po2::parse_args(params_.make_options_description(), args);

  size_t capacity = params_.num_procs;
  lc0_pool_.set_capacity(capacity);
}

}  // namespace a0achess
