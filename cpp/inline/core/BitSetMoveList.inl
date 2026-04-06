#include "core/BitSetMoveList.hpp"

namespace core {

namespace detail {

template <typename Move, typename InnerIt>
class BitSetMoveListIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = Move;
  using difference_type = std::ptrdiff_t;
  using pointer = const Move*;
  using reference = Move;

  BitSetMoveListIterator(InnerIt it) : it_(it) {}

  Move operator*() const { return Move(*it_); }

  BitSetMoveListIterator& operator++() {
    ++it_;
    return *this;
  }
  BitSetMoveListIterator operator++(int) {
    BitSetMoveListIterator temp = *this;
    ++it_;
    return temp;
  }
  bool operator==(const BitSetMoveListIterator& other) const { return it_ == other.it_; }
  bool operator!=(const BitSetMoveListIterator& other) const { return it_ != other.it_; }

 private:
  InnerIt it_;
};

}  // namespace detail

template <typename Move, int N>
Move BitSetMoveList<Move, N>::get_random(std::mt19937& prng) const {
  return Move(moves_.choose_random_on_index(prng));
}

template <typename Move, int N>
auto BitSetMoveList<Move, N>::begin() const {
  auto range = moves_.on_indices();
  using InnerIt = decltype(range.begin());
  return detail::BitSetMoveListIterator<Move, InnerIt>(range.begin());
}

template <typename Move, int N>
auto BitSetMoveList<Move, N>::end() const {
  auto range = moves_.on_indices();
  using InnerIt = decltype(range.end());
  return detail::BitSetMoveListIterator<Move, InnerIt>(range.end());
}

}  // namespace core
