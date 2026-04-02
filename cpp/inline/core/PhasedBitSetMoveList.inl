#include "core/PhasedBitSetMoveList.hpp"

namespace core {

namespace detail {

template <typename Move, typename InnerIt>
class PhasedBitSetMoveListIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = Move;
  using difference_type = std::ptrdiff_t;
  using pointer = const Move*;
  using reference = Move;

  PhasedBitSetMoveListIterator(InnerIt it, core::game_phase_t game_phase)
      : it_(it), game_phase_(game_phase) {}

  Move operator*() const { return Move(*it_, game_phase_); }
  bool operator==(const PhasedBitSetMoveListIterator&) const = default;
  bool operator!=(const PhasedBitSetMoveListIterator&) const = default;

  PhasedBitSetMoveListIterator& operator++() {
    ++it_;
    return *this;
  }
  PhasedBitSetMoveListIterator operator++(int) {
    PhasedBitSetMoveListIterator temp = *this;
    ++it_;
    return temp;
  }

 private:
  InnerIt it_;
  core::game_phase_t game_phase_;
};

}  // namespace detail

template <typename Move, int N>
void PhasedBitSetMoveList<Move, N>::add(const Move& move) {
  if (phase_ >= 0 && move.phase() != phase_) {
    throw util::Exception("move phase {} does not match PhasedBitSetMoveList phase {}",
                          move.phase(), phase_);
  }
  phase_ = move.phase();
  indices_.set(int(move.index()));
}

template <typename Move, int N>
void PhasedBitSetMoveList<Move, N>::remove(const Move& move) {
  if (phase_ < 0 || move.phase() != phase_) {
    throw util::Exception("move phase {} does not match PhasedBitSetMoveList phase {}",
                          move.phase(), phase_);
  }
  indices_.reset(int(move.index()));
}

template <typename Move, int N>
bool PhasedBitSetMoveList<Move, N>::contains(const Move& move) const {
  if (phase_ < 0 || move.phase() != phase_) {
    return false;
  }
  return indices_.test(int(move.index()));
}

template <typename Move, int N>
void PhasedBitSetMoveList<Move, N>::clear() {
  phase_ = -1;
  indices_.reset();
}

template <typename Move, int N>
Move PhasedBitSetMoveList<Move, N>::get_random(std::mt19937& prng) const {  // assumes !empty()
  if (phase_ < 0) {
    throw util::Exception("cannot get random move from empty PhasedBitSetMoveList");
  }
  int random_index = indices_.choose_random_on_index(prng);
  return Move(random_index, phase_);
}

template <typename Move, int N>
auto PhasedBitSetMoveList<Move, N>::begin() const {
  auto range = indices_.on_indices();
  using InnerIt = decltype(range.begin());
  return detail::PhasedBitSetMoveListIterator<Move, InnerIt>(range.begin(), phase_);
}

template <typename Move, int N>
auto PhasedBitSetMoveList<Move, N>::end() const {
  auto range = indices_.on_indices();
  using InnerIt = decltype(range.end());
  return detail::PhasedBitSetMoveListIterator<Move, InnerIt>(range.end(), phase_);
}

template <typename Move, int N>
size_t PhasedBitSetMoveList<Move, N>::serialize(char* buffer) const {
  // simply memcp the underlying bitset data, since it's already compact
  std::memcpy(buffer, this, sizeof(PhasedBitSetMoveList));
  return sizeof(PhasedBitSetMoveList);
}

template <typename Move, int N>
size_t PhasedBitSetMoveList<Move, N>::deserialize(const char* buffer) {
  std::memcpy(this, buffer, sizeof(PhasedBitSetMoveList));
  return sizeof(PhasedBitSetMoveList);
}

template <typename Move, int N>
std::string PhasedBitSetMoveList<Move, N>::to_string() const {
  // TODO: change this to include phase information as well
  // return std::format("phase: {}, indices: {}", phase_, indices_.to_string_natural());
  return indices_.to_string_natural();
}

}  // namespace core
