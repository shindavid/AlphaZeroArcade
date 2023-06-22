#include <othello/players/EdaxPlayer.hpp>

#include <string>

#include <boost/filesystem.hpp>

#include <util/BoostUtil.hpp>
#include <util/Config.hpp>

namespace othello {

inline auto EdaxPlayer::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("othello::EdaxPlayer options");
  return desc
      .template add_option<"depth", 'd'>
          (po::value<int>(&depth)->default_value(depth), "Search depth")
      ;
}

inline EdaxPlayer::EdaxPlayer(const Params& params) : params_(params) {
  line_buffer_.resize(1);
  std::string edax_dir_str = util::Config::instance()->get("othello.edax_dir", "");
  std::string edax_bin_str = util::Config::instance()->get("othello.edax_bin", "");

  if (edax_dir_str.empty()) {
    throw util::CleanException("othello.edax_dir not specified! Please follow setup instructions in py/othello/README.md");
  }
  if (edax_bin_str.empty()) {
    throw util::CleanException("othello.edax_bin not specified! Please follow setup instructions in py/othello/README.md");
  }
  boost::filesystem::path edax_dir(edax_dir_str);
  boost::filesystem::path edax_bin = edax_dir / edax_bin_str;
  if (!boost::filesystem::is_directory(edax_dir)) {
    throw util::Exception("Dir specified by config value 'othello.edax_dir' does not exist: %s. "
                          "Please follow setup instructions in py/othello/README.md", edax_dir.c_str());
  }
  if (!boost::filesystem::is_regular_file(edax_bin)) {
    throw util::Exception("File formed by combining config values 'othello.edax_dir' and 'othello.edax_bin' "
                          "does not exist: %s. Please follow setup instructions in py/othello/README.md",
                          edax_bin.c_str());
  }

  namespace bp = boost::process;
  proc_ = new bp::child(edax_bin_str, bp::start_dir(edax_dir), bp::std_out > out_, bp::std_err > bp::null,
                        bp::std_in < in_);

  std::string level_str = util::create_string("level %d\n", params_.depth);
  in_.write(level_str.c_str(), level_str.size());
  in_.flush();
}

inline void EdaxPlayer::start_game() {
  in_.write("i\n", 2);
  in_.flush();
}

inline void EdaxPlayer::receive_state_change(core::seat_index_t seat, const GameState&, core::action_index_t action) {
  if (seat == this->get_my_seat()) return;
  submit_action(action);
}

inline core::action_index_t EdaxPlayer::get_action(const GameState&, const ActionMask& valid_actions) {
  if (valid_actions.count() == 1) {  // only 1 possible move, no need to incur edax/IO overhead
    for (core::action_index_t action : bitset_util::on_indices(valid_actions)) {
      submit_action(action);
      return action;
    }
  }
  in_.write("go\n", 3);
  in_.flush();

  core::action_index_t action = -1;
  size_t n = 0;
  for (; std::getline(out_, line_buffer_[n]); ++n) {
    const std::string& line = line_buffer_[n];
    if (line.starts_with("Edax plays ")) {
      std::string move_str = line.substr(11, 2);
      if (move_str.starts_with("PS")) {  // "PS" is edax notation for pass
        action = kPass;
        break;
      } else {
        action = (move_str[0] - 'A') + 8 * (move_str[1] - '1');
        break;
      }
    }
    if (n + 1 == line_buffer_.size()) {
      line_buffer_.resize(line_buffer_.size() * 2);
    }
  }

  if (action < 0 || action >= kNumGlobalActions || !valid_actions[action]) {
    for (size_t i = 0; i < n; ++i) {
      std::cerr << line_buffer_[i] << std::endl;
    }
    throw util::Exception("EdaxPlayer::get_action: invalid action: %d", action);
  }
  return action;
}

inline void EdaxPlayer::submit_action(core::action_index_t action) {
  if (action == kPass) {
    in_.write("PS\n", 3);
    in_.flush();
  } else {
    char move_str[3];
    move_str[0] = char('A' + action % 8);
    move_str[1] = char('1' + action / 8);
    move_str[2] = '\n';
    in_.write(move_str, 3);
    in_.flush();
  }
}

}  // namespace othello
