#pragma once

#include <bitset>
#include <cstdint>
#include <iterator>

namespace c4 {

using column_t = int8_t;
const int kNumColumns = 7;
const int kNumRows = 6;

class ActionMask {
public:
    struct Iterator {
        Iterator(uint8_t mask) : mask_(mask) {}
        column_t operator*() const { return to_column();}
        Iterator& operator++() { mask_ -= (1 << to_column()); return *this; }
        Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }

    protected:
        column_t to_column() const { return __builtin_ctz(mask_); }
        uint8_t mask_;
    };

    Iterator begin() const { return Iterator(mask_); }
    Iterator end() const { return Iterator(0); }
    void set(column_t c) { mask_ |= (1<<c); }
    void unset(column_t c) { mask_ |= (1<<c); }
    bool get(column_t c) const { return mask_ & (1<<c); }

private:
    uint8_t mask_ = 0;
};

class GameState {
public:
    enum Color : uint8_t {
        eRed = 0,
        eYellow = 1,
        eNumColors = 2
    };

    GameState();

    Color get_current_player() const { return current_player_; }

private:
    uint64_t masks_[eNumColors] = {};
    Color current_player_ = eRed;
};

}  // namespace c4

#include <connect4/C4GameStateINLINES.cpp>
