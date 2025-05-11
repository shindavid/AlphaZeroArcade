#include <games/othello/players/EdaxPlayerGenerator.hpp>

#include <util/BoostUtil.hpp>

#include <format>

namespace othello {

inline std::string EdaxPlayerGenerator::get_default_name() const {
  return std::format("Edax-{}", params_.depth);
}

inline EdaxPlayerGenerator::Player* EdaxPlayerGenerator::generate(core::game_slot_index_t) {
  return new EdaxPlayer(&oracle_pool_, params_);
}

inline void EdaxPlayerGenerator::parse_args(const std::vector<std::string>& args) {
  namespace po2 = boost_util::program_options;
  po2::parse_args(params_.make_options_description(), args);

  // NOTE: if multiple Edax players are facing each other, they share the same OraclePool. In
  // this case, the capacity of the pool is set to the largest capacity of any of the
  // EdaxPlayerGenerator instances.
  size_t capacity = params_.num_oracle_procs;
  if (capacity > oracle_pool_.capacity()) {
    oracle_pool_.set_capacity(capacity);
  }
}

inline void EdaxPlayerGenerator::start_session() {
  if (oracle_pool_.capacity() > 0) {
    return;
  }

  // Capacity is not set. Default it to the number of game threads.
  oracle_pool_.set_capacity(server_->num_game_threads());
}

}  // namespace othello
