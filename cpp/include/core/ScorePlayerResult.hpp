#pragma once

#include "core/concepts/PlayerResultConcept.hpp"

#include <format>
#include <string>

namespace core {

/*
 * ScorePlayerResult is for games where each player's result is a continuous score (positive = won,
 * negative = lost). In a zero-sum game, scores across players sum to zero.
 *
 * Examples: poker (amount won/lost), other gambling games.
 */
struct ScorePlayerResult {
  float score;  // positive = won, negative = lost, zero = broke even

  std::string to_str() const { return std::format("{:.16g}", score); }
  bool is_win() const { return score > 0.0f; }
  bool is_loss() const { return score < 0.0f; }

  struct Aggregation {
    int win = 0;
    int loss = 0;
    int draw = 0;
    float total = 0.0f;

    void add(const ScorePlayerResult& r);
    std::string to_str() const;
  };
};

static_assert(core::concepts::PlayerResult<ScorePlayerResult>);

}  // namespace core

#include "inline/core/ScorePlayerResult.inl"
