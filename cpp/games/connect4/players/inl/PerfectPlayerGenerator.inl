#include <games/connect4/players/PerfectPlayerGenerator.hpp>

#include <util/BoostUtil.hpp>

namespace c4 {

inline std::string PerfectPlayerGenerator::get_default_name() const {
  return util::create_string("Perfect-%d", params_.strength);
}

inline void PerfectPlayerGenerator::parse_args(const std::vector<std::string>& args) {
  namespace po2 = boost_util::program_options;
  po2::parse_args(params_.make_options_description(), args);
}

}  // namespace c4
