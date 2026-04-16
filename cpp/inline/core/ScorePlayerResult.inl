#include "core/ScorePlayerResult.hpp"

namespace core {

inline void ScorePlayerResult::Aggregation::add(const ScorePlayerResult& r) {
  total += r.score;
  if (r.is_win())
    ++win;
  else if (r.is_loss())
    ++loss;
  else
    ++draw;
}

inline std::string ScorePlayerResult::Aggregation::to_str() const {
  return std::format("W{} L{} D{} [{:.16g}]", win, loss, draw, total);
}

}  // namespace core
