#pragma once

#include <cstdint>

#include <util/BitSet.hpp>

namespace c4 {

using column_t = int8_t;
const int kNumColumns = 7;
const int kNumRows = 6;

using ActionMask = util::BitSet<kNumColumns>;

class GameState {
public:
    enum Color : uint8_t {
        eRed = 0,
        eYellow = 1,
        eNumColors = 2
    };

    GameState();

    static int get_num_global_actions() { return kNumColumns; }

    Color get_current_player() const { return current_player_; }



private:
    uint64_t masks_[eNumColors] = {};
    Color current_player_ = eRed;
};

}  // namespace c4

#include <connect4/C4GameLogicINLINES.cpp>
