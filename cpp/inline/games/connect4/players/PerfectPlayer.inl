#include <games/connect4/players/PerfectPlayer.hpp>

#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/Exception.hpp>
#include <util/RepoUtil.hpp>
#include <util/StringUtil.hpp>

#include <boost/dll.hpp>

namespace c4 {

inline PerfectOracle::MoveHistory::MoveHistory() : char_pointer_(chars_) {}

inline PerfectOracle::MoveHistory::MoveHistory(const MoveHistory& history) {
  memcpy(chars_, history.chars_, sizeof(chars_));
  char_pointer_ = chars_ + (history.char_pointer_ - history.chars_);
}

inline void PerfectOracle::MoveHistory::reset() {
  char_pointer_ = chars_;
  *char_pointer_ = 0;
}

inline void PerfectOracle::MoveHistory::append(int move) {
  *(char_pointer_++) = char(int('1') + move);  // connect4 program uses 1-indexing
}

inline std::string PerfectOracle::MoveHistory::to_string() const {
  return std::string(chars_, char_pointer_ - chars_);
}

inline void PerfectOracle::MoveHistory::write(boost::process::opstream& in) {
  *char_pointer_ = '\n';
  in.write(chars_, char_pointer_ - chars_ + 1);
  in.flush();
}

inline PerfectOracle::PerfectOracle() {
  auto extra_dir = boost::dll::program_location().parent_path() / "extra";
  auto c4_solver_bin = extra_dir / "c4solver";
  auto c4_solver_book = extra_dir / "7x6.book";

  for (const auto& path : {c4_solver_bin, c4_solver_book}) {
    if (!boost::filesystem::is_regular_file(path)) {
      throw util::CleanException("File does not exist: %s", path.c_str());
    }
  }

  namespace bp = boost::process;
  std::string c4_cmd =
      util::create_string("%s -b %s -a", c4_solver_bin.c_str(), c4_solver_book.c_str());
  proc_ = new bp::child(c4_cmd, bp::std_out > out_, bp::std_err > bp::null, bp::std_in < in_);
}

inline PerfectOracle* PerfectOracle::get_instance() {
  std::unique_lock lock(static_mutex_);
  if (oracles_.empty() || oracles_.back()->client_count_ >= kNumClientsPerOracle) {
    oracles_.push_back(new PerfectOracle());
  }
  auto oracle = oracles_.back();
  oracle->client_count_++;
  return oracle;
}

inline PerfectOracle::~PerfectOracle() { delete proc_; }

inline auto PerfectPlayer::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("c4::PerfectPlayer options");
  return desc
      .template add_option<"strength", 's'>(
          po::value<int>(&strength)->default_value(strength),
          "strength (0-21). The last s moves are played perfectly, the others randomly. 0 is "
          "random, 21 is perfect.")
      .template add_option<"verbose", 'v'>(po::bool_switch(&verbose)->default_value(verbose),
                                           "verbose mode");
}

inline PerfectPlayer::PerfectPlayer(const Params& params) : params_(params) {
  oracle_ = PerfectOracle::get_instance();
  util::clean_assert(params_.strength >= 0 && params_.strength <= 21,
                     "strength must be in [0, 21]");
}

inline void PerfectPlayer::start_game() { move_history_.reset(); }

inline void PerfectPlayer::receive_state_change(core::seat_index_t, const GameState&,
                                                const Action& action) {
  move_history_.append(action[0]);
}

}  // namespace c4
