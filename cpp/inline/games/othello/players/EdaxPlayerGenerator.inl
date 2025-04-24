#include <games/othello/players/EdaxPlayerGenerator.hpp>

#include <core/Globals.hpp>
#include <util/BoostUtil.hpp>

#include <format>

namespace othello {

inline std::string EdaxPlayerGenerator::get_default_name() const {
  return std::format("Edax-{}", params_.depth);
}

inline EdaxPlayerGenerator::Player* EdaxPlayerGenerator::generate(core::game_slot_index_t) {
  return new EdaxPlayer(&oracle_pool_, params_);
}

inline void EdaxPlayerGenerator::start_session() {
  int num_game_threads = core::Globals::num_game_threads;
  int capacity = params_.num_oracle_procs;
  if (capacity <= 0) {
    capacity = num_game_threads;
  }
  oracle_pool_.set_capacity(capacity);
}

inline void EdaxPlayerGenerator::parse_args(const std::vector<std::string>& args) {
  namespace po2 = boost_util::program_options;
  po2::parse_args(params_.make_options_description(), args);
}

}  // namespace othello
