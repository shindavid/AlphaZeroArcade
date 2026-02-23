#pragma once

#include "games/chess/Types.hpp"

namespace chess {

constexpr int kMoveHistory = 8;
constexpr int kPlanesPerBoard = 13;
constexpr int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;

enum class FillEmptyHistory { NO, FEN_ONLY, ALWAYS };

uint16_t MoveToNNIndex(Move move, int transform);
Move MoveFromNNIndex(int idx, int transform);

}  // namespace chess

#include "inline/games/chess/Encoder.inl"
