#include "core/WinLossDrawPlayerResult.hpp"

#include <array>
#include <format>

namespace core {

inline std::string WinLossDrawPlayerResult::to_str() const {
  if (kind == kWin)
    return "W";
  else if (kind == kLoss)
    return "L";
  else
    return "D";
}

template <int NumPlayers>
inline auto WinLossDrawPlayerResult::make_win(core::seat_index_t seat) {
  std::array<WinLossDrawPlayerResult, NumPlayers> outcome;
  for (int s = 0; s < NumPlayers; ++s) {
    outcome[s].kind = (s == seat) ? kWin : kLoss;
  }
  return outcome;
}

template <int NumPlayers>
inline auto WinLossDrawPlayerResult::make_draw() {
  std::array<WinLossDrawPlayerResult, NumPlayers> outcome;
  for (int s = 0; s < NumPlayers; ++s) {
    outcome[s].kind = kDraw;
  }
  return outcome;
}

inline void WinLossDrawPlayerResult::Aggregation::add(const WinLossDrawPlayerResult& r) {
  if (r.kind == kWin)
    ++win;
  else if (r.kind == kLoss)
    ++loss;
  else
    ++draw;
}

inline std::string WinLossDrawPlayerResult::Aggregation::to_str() const {
  return std::format("W{} L{} D{} [{:.16g}]", win, loss, draw, win + 0.5f * draw);
}

}  // namespace core
