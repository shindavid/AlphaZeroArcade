#include <games/blokus/players/HumanTuiPlayer.hpp>

#include <games/blokus/Constants.hpp>
#include <util/BitSet.hpp>

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace blokus {

inline void HumanTuiPlayer::start_game() {
  pending_poc_ = -1;
  passed_ = false;
}

inline core::action_t HumanTuiPlayer::prompt_for_action(const FullState& state,
                                                        const ActionMask& valid_actions) {
  if (passed_) {
    // no need to keep prompting for moves if we've passed already
    return kPass;
  }

  if (pending_poc_ >= 0) {
    PieceOrientationCorner poc(pending_poc_);
    pending_poc_ = -1;
    return poc.to_action();
  } else {
    if (valid_actions.size() == 1) {
      return prompt_for_pass();
    }

    p_map_t p_map;
    load_actions(p_map, state, valid_actions);

    Piece p(-1);
    PieceOrientation po(-1);
    Location root_loc(-1, -1);

    while (true) {
      if (!prompt_for_piece(p_map, p)) continue;
      if (!prompt_for_orientation(p_map, p, po)) continue;
      if (!prompt_for_root_location(p_map, p, po, root_loc)) continue;
      break;
    }

    value_t value = p_map[p][po][root_loc.flatten()];
    pending_poc_ = value.poc;
    return value.loc.flatten();
  }
}

inline bool HumanTuiPlayer::prompt_for_piece(const p_map_t& p_map, Piece& p) const {
  if (p >= 0) return true;
  throw std::runtime_error("Not implemented");
}

inline bool HumanTuiPlayer::prompt_for_orientation(const p_map_t&, Piece,
                                                   PieceOrientation& po) const {
  if (po >= 0) return true;
  throw std::runtime_error("Not implemented");
}

inline bool HumanTuiPlayer::prompt_for_root_location(const p_map_t&, Piece, PieceOrientation,
                                                     Location& root_loc) const {
  throw std::runtime_error("Not implemented");
}

inline core::action_t HumanTuiPlayer::prompt_for_pass() {
  std::cout << "No moves available. Press Enter to pass. ";
  std::cout.flush();

  std::string input;
  std::getline(std::cin, input);
  passed_ = true;
  return kPass;
}

inline void HumanTuiPlayer::load_actions(p_map_t& p_map, const FullState& state,
                                         const ActionMask& valid_actions) const {
  for (core::action_t action : bitset_util::on_indices(valid_actions)) {
    util::release_assert(action < kPass);
    Location loc = Location::unflatten(action);

    FullState copy = state;
    Game::Rules::apply(copy, action);
    ActionMask valid_actions2 = Game::Rules::get_legal_moves(copy);
    util::release_assert(valid_actions2.any());
    for (core::action_t action2 : bitset_util::on_indices(valid_actions)) {
      util::release_assert(action2 > kPass);
      PieceOrientationCorner poc = PieceOrientationCorner::from_action(action2);
      Piece p = poc.to_piece();
      PieceOrientation po = poc.to_piece_orientation();
      Location loc2 = poc.get_root_location(loc);
      p_map[p][po][loc2.flatten()] = {poc, loc};
    }
  }
}

}  // namespace blokus
