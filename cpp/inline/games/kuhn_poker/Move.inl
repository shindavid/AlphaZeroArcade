#include "games/kuhn_poker/Move.hpp"

#include "games/kuhn_poker/Constants.hpp"
#include "util/Exceptions.hpp"

namespace kuhn_poker {

inline std::string Move::to_str() const {
  if (phase_ == kDealPhase) {
    return std::format("d{}", index_);
  } else {
    return kActionNames[index_];
  }
}

inline Move Move::from_str(const InfoSetState&, std::string_view s) {
  if (s[0] == 'd') {
    int index = s[1] - '0';
    return Move(index, kDealPhase);
  }
  for (int i = 0; i < kNumBettingActions; ++i) {
    if (s == kActionNames[i]) {
      return Move(i, kBettingPhase);
    }
  }
  throw util::Exception("Invalid move string: {}", std::string(s));
}

}  // namespace kuhn_poker
