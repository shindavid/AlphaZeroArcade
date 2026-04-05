#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/PlayerResultConcept.hpp"

#include <string>

namespace core {

struct WinLossDrawPlayerResult {
  enum Kind { kWin, kLoss, kDraw };
  Kind kind;

  std::string to_str() const;
  bool is_win() const { return kind == kWin; }
  bool is_loss() const { return kind == kLoss; }

  template <int NumPlayers>
  static auto make_win(core::seat_index_t seat);

  template <int NumPlayers>
  static auto make_draw();

  struct Aggregation {
    int win = 0;
    int loss = 0;
    int draw = 0;

    void add(const WinLossDrawPlayerResult& r);
    std::string to_str() const;
  };
};

static_assert(core::concepts::PlayerResult<WinLossDrawPlayerResult>);

}  // namespace core

#include "inline/core/WinLossDrawPlayerResult.inl"
