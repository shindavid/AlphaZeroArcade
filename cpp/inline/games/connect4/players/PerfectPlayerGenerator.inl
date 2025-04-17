#include <games/connect4/players/PerfectPlayerGenerator.hpp>

#include <util/BoostUtil.hpp>

namespace c4 {

inline std::string PerfectPlayerGenerator::get_default_name() const {
  return util::create_string("Perfect-%d", params_.strength);
}

inline core::AbstractPlayer<c4::Game>* PerfectPlayerGenerator::generate(core::game_thread_id_t) {
  if (!pool_initialized_) {
    oracle_pool_.set_capacity(params_.num_oracle_procs);
    pool_initialized_ = true;
  }
  return new PerfectPlayer(&oracle_pool_, params_);
}

inline void PerfectPlayerGenerator::parse_args(const std::vector<std::string>& args) {
  namespace po2 = boost_util::program_options;
  po2::parse_args(params_.make_options_description(), args);
}

}  // namespace c4
