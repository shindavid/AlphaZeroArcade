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
  base_t::start_game();
}

inline core::action_t HumanTuiPlayer::prompt_for_action(const BaseState& state,
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
    if (valid_actions[kPass]) {
      return prompt_for_pass();
    }

    p_map_t p_map;
    load_actions(p_map, state, valid_actions);

    Piece p(-1);
    PieceOrientation po(-1);
    Location root_loc(-1, -1);

    bool reprint_state = false;
    while (true) {
      if (reprint_state) {
        util::clearscreen();
        this->print_state(state, false);
      }
      reprint_state = true;
      prompt_for_piece(state, p_map, p);
      if (!prompt_for_orientation(state, p_map, p, po)) continue;
      if (!prompt_for_root_location(state, p_map, p, po, root_loc)) continue;
      break;
    }

    value_t value = p_map[p][po][root_loc.flatten()];
    pending_poc_ = value.poc;
    return value.loc.flatten();
  }
}

inline void HumanTuiPlayer::prompt_for_piece(const BaseState& state, const p_map_t& p_map,
                                             Piece& p) {
  if (p >= 0) return;

  TuiPrompt prompt;
  for (auto [pi, _] : p_map) {
    Piece candidate_p(pi);
    candidate_p.write_to(prompt, this->get_my_seat());
  }
  prompt.print();

  while (true) {
    std::cout << "Choose a piece: ";
    std::string response;
    std::getline(std::cin, response);
    try {
      p = std::stoi(response);
      if (p_map.contains(p)) break;
    } catch (...) {
    }
    std::cout << "Invalid input!" << std::endl;
  }
}

inline bool HumanTuiPlayer::prompt_for_orientation(const BaseState& state, const p_map_t& p_map,
                                                   Piece& p, PieceOrientation& po) {
  if (po >= 0) return true;

  TuiPrompt prompt;
  const po_map_t& po_map = p_map.at(p);
  int label = 0;
  piece_orientation_index_t candidates[8];
  for (auto [poi, _] : po_map) {
    candidates[label] = poi;
    PieceOrientation candidate_po(poi);
    candidate_po.write_to(prompt, this->get_my_seat(), label);
    label++;
  }

  int num_labels = label;

  prompt.print();
  while (true) {
    std::cout << "Choose an orientation (Enter to undo): ";
    std::string response;
    std::getline(std::cin, response);
    if (response.empty()) {
      p = -1;
      return false;
    }
    try {
      po = std::stoi(response);
      if (po >= 0 && po < num_labels) {
        po = candidates[po];
        break;
      }
    } catch (...) {
    }
    std::cout << "Invalid input!" << std::endl;
  }
  return true;
}

inline bool HumanTuiPlayer::prompt_for_root_location(const BaseState& state, const p_map_t& p_map,
                                                     Piece& p, PieceOrientation& po,
                                                     Location& root_loc) {
  const po_map_t& po_map = p_map.at(p);
  const inner_map_t& inner_map = po_map.at(po);

  color_t color = this->get_my_seat();
  PieceOrientationCorner poc = po.canonical_corner();

  std::cout << "Selected piece:" << std::endl;
  poc.pretty_print(std::cout, color);

  while (true) {
    std::cout << "Choose a location (Enter to undo): ";
    std::string response;
    std::getline(std::cin, response);
    if (response.empty()) {
      p = -1;
      po = -1;
      return false;
    }
    Location loc = Location::from_string(response);
    if (loc.valid()) {
      root_loc = poc.get_root_location(loc);
      int flattened = root_loc.flatten();

      if (inner_map.contains(flattened)) {
        return true;
      }
    }
    std::cout << "Invalid input!" << std::endl;
  }
  return true;
}

inline core::action_t HumanTuiPlayer::prompt_for_pass() {
  std::cout << "No moves available. Press Enter to pass. ";
  std::cout.flush();

  std::string input;
  std::getline(std::cin, input);
  passed_ = true;
  return kPass;
}

inline void HumanTuiPlayer::load_actions(p_map_t& p_map, const BaseState& state,
                                         const ActionMask& valid_actions) const {
  for (core::action_t action : bitset_util::on_indices(valid_actions)) {
    // std::cout << "DBG action=" << int(action) << std::endl;
    util::release_assert(action < kPass);
    Location loc = Location::unflatten(action);
    // std::cout << "DBG loc=" << loc.to_string() << std::endl;

    StateHistory history;
    history.update(state);
    Game::Rules::apply(history, action);
    ActionMask valid_actions2 = Game::Rules::get_legal_moves(history);
    util::release_assert(valid_actions2.any());
    for (core::action_t action2 : bitset_util::on_indices(valid_actions2)) {
      // std::cout << "DBG   action2=" << int(action2) << std::endl;
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
