#pragma once

#include "core/concepts/PlayerResultConcept.hpp"

#include <format>
#include <string>

namespace core {

struct WinSharePlayerResult {
  float share;  // fraction of the win attributed to this player, in [0, 1]

  std::string to_str() const { return std::format("{:.16g}", share); }
  bool is_win() const { return share == 1.0f; }
  bool is_loss() const { return share == 0.0f; }

  struct Aggregation {
    int win = 0;   // share == 1.0
    int loss = 0;  // share == 0.0
    int draw = 0;  // share in (0, 1)
    float total = 0.0f;

    void add(const WinSharePlayerResult& r);
    std::string to_str() const;
  };
};

static_assert(core::concepts::PlayerResult<WinSharePlayerResult>);

}  // namespace core

#include "inline/core/WinSharePlayerResult.inl"
