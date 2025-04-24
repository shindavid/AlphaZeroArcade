#include <stdexcept>
#include <util/StaticCircularBuffer.hpp>

namespace util {

template <typename T, int N>
void StaticCircularBuffer<T, N>::push_back(const T& value) {
  buffer_[(start_ + count_) % N] = value;
  if (count_ < N) {
    ++count_;
  } else {
    start_ = (start_ + 1) % N;  // overwrite oldest
  }
}

template <typename T, int N>
void StaticCircularBuffer<T, N>::pop_back() {
  if (empty()) throw std::out_of_range("StaticCircularBuffer: pop_back() on empty buffer");
  --count_;
}

template <typename T, int N>
T& StaticCircularBuffer<T, N>::back() {
  if (empty()) throw std::out_of_range("StaticCircularBuffer: back() on empty buffer");
  return buffer_[(start_ + count_ - 1) % N];
}

template <typename T, int N>
const T& StaticCircularBuffer<T, N>::back() const {
  if (empty()) throw std::out_of_range("StaticCircularBuffer: back() on empty buffer");
  return buffer_[(start_ + count_ - 1) % N];
}

template <typename T, int N>
void StaticCircularBuffer<T, N>::clear() {
  start_ = 0;
  count_ = 0;
}

template <typename T, int N>
T& StaticCircularBuffer<T, N>::operator[](std::size_t i) {
  std::size_t k = start_ + i;
  return buffer_[k - (k >= N ? N : 0)];
}

template <typename T, int N>
const T& StaticCircularBuffer<T, N>::operator[](std::size_t i) const {
  std::size_t k = start_ + i;
  return buffer_[k - (k >= N ? N : 0)];
}

}  // namespace util
