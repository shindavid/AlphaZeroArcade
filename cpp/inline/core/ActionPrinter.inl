#include "include/core/ActionPrinter.hpp"

namespace core {

template <concepts::Game Game>
ActionPrinter<Game>::ActionPrinter(const MoveSet& moves) {
  int i = 0;
  for (Move move : moves) {
    array_[i] = i;
    moves_[i] = move;
    i++;
  }
}

template <concepts::Game Game>
void ActionPrinter<Game>::update_format_map(eigen_util::PrintArrayFormatMap& fmt_map) const {
  fmt_map["action"] = [&](float x, int) { return moves_[int(x)].to_str(); };
}

}  // namespace core
