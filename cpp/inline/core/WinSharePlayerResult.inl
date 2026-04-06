#include "core/WinSharePlayerResult.hpp"

namespace core {

inline void WinSharePlayerResult::Aggregation::add(const WinSharePlayerResult& r) {
  total += r.share;
  if (r.is_win())
    ++win;
  else if (r.is_loss())
    ++loss;
  else
    ++draw;
}

inline std::string WinSharePlayerResult::Aggregation::to_str() const {
  // NOTE: W/L/D format required for compatibility with ratings.py extract_match_record(),
  // which expects the format "W%d L%d D%d". Here W=share==1.0, L=share==0.0, D=share in (0,1).
  // Can be changed once Python-side parsing is made more robust.
  return std::format("W{} L{} D{} [{:.16g}]", win, loss, draw, total);
}

}  // namespace core
