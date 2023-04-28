#include <othello/GameState.hpp>

#include <bit>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <util/AnsiCodes.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>

inline std::size_t std::hash<othello::GameState>::operator()(const othello::GameState& state) const {
  return state.hash();
}

namespace othello {

inline size_t GameState::serialize_action(char* buffer, size_t buffer_size, common::action_index_t action) {
  size_t n = snprintf(buffer, buffer_size, "%c%d", 'A' + (action % kBoardDimension), 1 + (action / kBoardDimension));
  if (n >= buffer_size) {
    throw util::Exception("Buffer too small (%ld >= %ld)", n, buffer_size);
  }
  return n;
}

inline void GameState::deserialize_action(const char* buffer, common::action_index_t* action) {
  int col = buffer[0] - 'A';
  int row = boost::lexical_cast<int>(buffer + 1) - 1;

  if (col < 0 || col >= kBoardDimension || row < 0 || row >= kBoardDimension) {
    throw util::Exception("Invalid action \"%s\" (col=%d, row=%d)", buffer, col, row);
  }
  *action = row * kBoardDimension + col;
}

inline size_t GameState::serialize_state_change(
    char* buffer, size_t buffer_size, common::seat_index_t seat, common::action_index_t action) const
{
  return serialize_action(buffer, buffer_size, action);
}

inline void GameState::deserialize_state_change(
    const char* buffer, common::seat_index_t* seat, common::action_index_t* action)
{
  *seat = get_current_player();
  deserialize_action(buffer, action);
  apply_move(*action);
}

inline size_t GameState::serialize_game_end(char* buffer, size_t buffer_size, const GameOutcome& outcome) const {
  size_t n = 0;
  bool b = outcome[kBlack] > 0;
  bool w = outcome[kWhite] > 0;
  n += snprintf(buffer + n, buffer_size - n, b ? "B" : "");
  n += snprintf(buffer + n, buffer_size - n, w ? "W" : "");
  if (n >= buffer_size) {
    throw util::Exception("Buffer too small (%ld >= %ld)", n, buffer_size);
  }
  return n;
}

inline void GameState::deserialize_game_end(const char* buffer, GameOutcome* outcome) {
  outcome->setZero();
  const char* c = buffer;
  while (*c != '\0') {
    switch (*c) {
      case 'B': (*outcome)(kBlack) = 1; break;
      case 'W': (*outcome)(kWhite) = 1; break;
      default: throw util::Exception(R"(Invalid game end "%c" parsed from "%s")", *c, buffer);
    }
    ++c;
  }

  *outcome /= outcome->sum();
}

inline common::GameStateTypes<GameState>::GameOutcome GameState::apply_move(common::action_index_t action) {
  throw util::Exception("TODO");
}

inline GameState::ActionMask GameState::get_valid_actions() const {
  throw util::Exception("TODO");
}

template<eigen_util::FixedTensorConcept InputSlab> void GameState::tensorize(InputSlab& tensor) const {
  throw util::Exception("TODO");
}

inline void GameState::dump(common::action_index_t last_action, const player_name_array_t* player_names) const {
  throw util::Exception("TODO");
}

inline void GameState::row_dump(row_t row, column_t blink_column) const {
  throw util::Exception("TODO");
}

inline bool GameState::operator==(const GameState& other) const {
  return full_mask_ == other.full_mask_ && cur_player_mask_ == other.cur_player_mask_ && cur_player_ == other.cur_player_;
}

inline common::action_index_t GameState::prompt_for_action() {
  std::cout << "Enter move [A1-H7]: ";
  std::cout.flush();
  std::string input;
  std::getline(std::cin, input);
  if (input.size() < 2) {
    return -1;
  }

  int col = input[0] - 'A';
  int row;
  try {
    row = std::stoi(input.substr(1)) - 1;
  } catch (std::invalid_argument& e) {
    return -1;
  } catch (std::out_of_range& e) {
    return -1;
  }

  if (col < 0 || col >= kBoardDimension || row < 0 || row >= kBoardDimension) {
    return -1;
  }
  return row * kBoardDimension + col;
}

inline void GameState::dump_mcts_output(
    const ValueProbDistr& mcts_value, const LocalPolicyProbDistr& mcts_policy, const MctsResults& results)
{
  throw util::Exception("TODO");
}

}  // namespace othello
