#include "core/WinLossPlayerResult.hpp"

#include <array>
#include <format>

namespace core {

inline std::string WinLossPlayerResult::to_str() const {
  if (kind == kWin)
    return "W";
  else
    return "L";
}

template <int NumPlayers>
inline auto WinLossPlayerResult::make_win(core::seat_index_t seat) {
  std::array<WinLossPlayerResult, NumPlayers> outcome;
  for (int s = 0; s < NumPlayers; ++s) {
    outcome[s].kind = (s == seat) ? kWin : kLoss;
  }
  return outcome;
}

inline void WinLossPlayerResult::Aggregation::add(const WinLossPlayerResult& r) {
  if (r.kind == kWin)
    ++win;
  else
    ++loss;
}

inline std::string WinLossPlayerResult::Aggregation::to_str() const {
  // NOTE: D0 is a dummy field required for compatibility with ratings.py
  // extract_match_record(), which expects the format "W%d L%d D%d". Can be removed once
  // Python-side parsing is made more robust.
  return std::format("W{} L{} D0 [{}]", win, loss, win);
}

}  // namespace core
