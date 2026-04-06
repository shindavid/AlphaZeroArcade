#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/PlayerResultConcept.hpp"

#include <string>

namespace core {

struct WinLossPlayerResult {
  enum Kind { kWin, kLoss };
  Kind kind;

  std::string to_str() const;
  bool is_win() const { return kind == kWin; }
  bool is_loss() const { return kind == kLoss; }

  template <int NumPlayers>
  static auto make_win(core::seat_index_t seat);

  struct Aggregation {
    int win = 0;
    int loss = 0;

    void add(const WinLossPlayerResult& r);
    std::string to_str() const;
  };
};

static_assert(core::concepts::PlayerResult<WinLossPlayerResult>);

}  // namespace core

#include "inline/core/WinLossPlayerResult.inl"
