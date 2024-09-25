#pragma once

#include <core/BasicTypes.hpp>
#include <generic_players/HumanTuiPlayer.hpp>
#include <games/blokus/Game.hpp>

#include <map>
#include <vector>

namespace blokus {

/*
 * blokus::Game splits actions into 2 piec  followed by a pie  * awkward for humans.
 *
 * This class's prompt_for_action() method prompts the user in a more human-friendly way: first,
 * it asks for a piece, then it asks for an orientation, and finally it asks for a board location.
 */
class HumanTuiPlayer : public generic::HumanTuiPlayer<Game> {
 public:
  using base_t = generic::HumanTuiPlayer<Game>;
  void start_game() override;

 private:
  #pragma pack(push, 1)
  struct value_t {
    piece_orientation_corner_index_t poc;
    Location loc;
  };
  #pragma pack(pop)

  using BaseState = Game::BaseState;
  using StateHistory = Game::StateHistory;
  using ActionMask = Game::Types::ActionMask;
  using flat_location_t = int;
  using inner_map_t = std::map<flat_location_t, value_t>;
  using po_map_t = std::map<piece_orientation_index_t, inner_map_t>;
  using p_map_t = std::map<piece_index_t, po_map_t>;

  core::action_t prompt_for_action(const BaseState&, const ActionMask&) override;
  void prompt_for_piece(const BaseState&, const p_map_t&, Piece&);
  bool prompt_for_orientation(const BaseState&, const p_map_t&, Piece&, PieceOrientation&);
  bool prompt_for_root_location(const BaseState&, const p_map_t&, Piece&, PieceOrientation&,
                                Location& root_loc);
  core::action_t prompt_for_pass();
  void load_actions(p_map_t&, const BaseState&, const ActionMask&) const;

  piece_orientation_corner_index_t pending_poc_;
  bool passed_;
};

}  // namespace blokus

#include <inline/games/blokus/players/HumanTuiPlayer.inl>
