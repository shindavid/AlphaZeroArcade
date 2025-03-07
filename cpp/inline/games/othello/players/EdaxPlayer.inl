#include <games/othello/players/EdaxPlayer.hpp>

#include <string>

#include <boost/dll.hpp>
#include <boost/filesystem.hpp>

#include <util/BoostUtil.hpp>

namespace othello {

inline auto EdaxPlayer::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("othello::EdaxPlayer options");
  return desc
      .template add_option<"depth", 'd'>(po::value<int>(&depth)->default_value(depth),
                                         "Search depth")
      .template add_option<"verbose", 'v'>(po::bool_switch(&verbose)->default_value(verbose),
                                           "edax player verbose mode");
}

inline EdaxPlayer::EdaxPlayer(const Params& params) : params_(params) {
  line_buffer_.resize(1);

  auto edax_dir = boost::filesystem::path("extra_deps/edax-reversi");
  auto edax_bin = edax_dir / "bin" / "lEdax-x64-modern";

  if (!boost::filesystem::is_regular_file(edax_bin)) {
    throw util::CleanException("File does not exist: %s", edax_bin.c_str());
  }

  namespace bp = boost::process;
  proc_ = new bp::child(edax_bin.c_str(), bp::start_dir(edax_dir), bp::std_out > out_,
                        bp::std_err > bp::null, bp::std_in < in_);

  std::string level_str = util::create_string("level %d\n", params_.depth);
  in_.write(level_str.c_str(), level_str.size());
  in_.flush();
}

inline void EdaxPlayer::start_game() {
  in_.write("i\n", 2);
  in_.flush();
}

inline void EdaxPlayer::receive_state_change(core::seat_index_t seat, const State&,
                                             core::action_t action) {
  if (seat == this->get_my_seat()) return;
  submit_action(action);
}

inline EdaxPlayer::ActionResponse EdaxPlayer::get_action_response(const ActionRequest& request) {
  const ActionMask& valid_actions = request.valid_actions;

  int num_valid_actions = valid_actions.count();
  if (params_.verbose) {
    std::cout << "EdaxPlayer::get_action_response() - num_valid_actions=" << num_valid_actions
              << std::endl;
  }
  if (num_valid_actions == 1) {  // only 1 possible move, no need to incur edax/IO overhead
    core::action_t action = bitset_util::get_nth_on_index(valid_actions, 0);
    submit_action(action);
    return action;
  }
  in_.write("go\n", 3);
  in_.flush();

  int a = -1;
  size_t n = 0;
  for (; std::getline(out_, line_buffer_[n]); ++n) {
    const std::string& line = line_buffer_[n];
    if (line.starts_with("Edax plays ")) {
      if (params_.verbose) {
        std::cout << line << std::endl;
      }
      std::string move_str = line.substr(11, 2);
      if (move_str.starts_with("PS")) {  // "PS" is edax notation for pass
        a = kPass;
        break;
      } else {
        a = (move_str[0] - 'A') + 8 * (move_str[1] - '1');
        break;
      }
    }
    if (n + 1 == line_buffer_.size()) {
      line_buffer_.resize(line_buffer_.size() * 2);
    }
  }

  if (a < 0 || a >= kNumGlobalActions || !valid_actions[a]) {
    for (size_t i = 0; i < n; ++i) {
      std::cerr << line_buffer_[i] << std::endl;
    }
    throw util::Exception("EdaxPlayer::get_action_response: invalid action: %d", a);
  }

  return a;
}

inline void EdaxPlayer::submit_action(const core::action_t action) {
  int a = action;
  if (a == kPass) {
    if (params_.verbose) {
      std::cout << "EdaxPlayer::submit_action() - PS" << std::endl;
    }

    in_.write("PS\n", 3);
    in_.flush();
  } else {
    char move_str[3];
    move_str[0] = char('A' + a % 8);
    move_str[1] = char('1' + a / 8);
    move_str[2] = '\n';
    in_.write(move_str, 3);
    in_.flush();
    if (params_.verbose) {
      std::cout << "EdaxPlayer::submit_action() - " << move_str[0] << move_str[1] << std::endl;
    }
  }
}

}  // namespace othello
