#pragma once

#include <array>
#include <cstddef>

namespace util {

/*
 * StaticCircularBuffer<T, N> is a circular buffer of size N that stores T objects. It is a
 * fixed-size buffer, meaning that it does not dynamically allocate memory. By contrast,
 * boost::circular_buffer<T> is a dynamic buffer that can grow and shrink in size.
 */
template <typename T, int N>
class StaticCircularBuffer {
 public:
  StaticCircularBuffer() : start_(0), count_(0) {}

  bool empty() const { return count_ == 0; }
  std::size_t size() const { return count_; }

  void push_back(const T& value);
  void push_front(const T& value);
  void pop_back();
  T& back();
  const T& back() const;
  T& front();
  const T& front() const;
  void clear();

  class iterator {
   public:
    using value_type = T;
    using reference = T&;
    using pointer = T*;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    iterator() = default;
    iterator(const iterator&) = default;
    iterator& operator=(const iterator&) = default;

    iterator(StaticCircularBuffer* buf, std::size_t pos) : buf_(buf), pos_(pos) {}

    pointer operator->() const { return &(*(*this)); }
    reference operator*() const { return (*buf_)[pos_]; }
    reference operator[](difference_type n) const { return (*buf_)[pos_ + n]; }

    iterator& operator++() {
      ++pos_;
      return *this;
    }
    iterator operator++(int) {
      iterator tmp = *this;
      ++pos_;
      return tmp;
    }

    iterator& operator--() {
      --pos_;
      return *this;
    }
    iterator operator--(int) {
      iterator tmp = *this;
      --pos_;
      return tmp;
    }

    iterator& operator+=(difference_type n) {
      pos_ += n;
      return *this;
    }
    iterator& operator-=(difference_type n) {
      pos_ -= n;
      return *this;
    }

    iterator operator+(difference_type n) const { return iterator(buf_, pos_ + n); }
    friend iterator operator+(difference_type n, const iterator& it) { return it + n; }
    iterator operator-(difference_type n) const { return iterator(buf_, pos_ - n); }
    difference_type operator-(const iterator& other) const { return pos_ - other.pos_; }

    bool operator==(const iterator& other) const { return pos_ == other.pos_; }
    bool operator!=(const iterator& other) const { return pos_ != other.pos_; }
    bool operator<(const iterator& other) const { return pos_ < other.pos_; }
    bool operator>(const iterator& other) const { return pos_ > other.pos_; }
    bool operator<=(const iterator& other) const { return pos_ <= other.pos_; }
    bool operator>=(const iterator& other) const { return pos_ >= other.pos_; }

   private:
    StaticCircularBuffer* buf_ = nullptr;
    std::size_t pos_ = 0;
  };

  class const_iterator {
   public:
    using value_type = T;
    using reference = const T&;
    using pointer = const T*;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    const_iterator() = default;
    const_iterator(const const_iterator&) = default;
    const_iterator& operator=(const const_iterator&) = default;

    const_iterator(const StaticCircularBuffer* buf, std::size_t pos) : buf_(buf), pos_(pos) {}

    pointer operator->() const { return &(*(*this)); }
    reference operator*() const { return (*buf_)[pos_]; }
    reference operator[](difference_type n) const { return (*buf_)[pos_ + n]; }

    const_iterator& operator++() {
      ++pos_;
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++pos_;
      return tmp;
    }

    const_iterator& operator--() {
      --pos_;
      return *this;
    }
    const_iterator operator--(int) {
      const_iterator tmp = *this;
      --pos_;
      return tmp;
    }

    const_iterator& operator+=(difference_type n) {
      pos_ += n;
      return *this;
    }
    const_iterator& operator-=(difference_type n) {
      pos_ -= n;
      return *this;
    }

    const_iterator operator+(difference_type n) const { return const_iterator(buf_, pos_ + n); }
    friend const_iterator operator+(difference_type n, const const_iterator& it) { return it + n; }
    const_iterator operator-(difference_type n) const { return const_iterator(buf_, pos_ - n); }
    difference_type operator-(const const_iterator& other) const { return pos_ - other.pos_; }

    bool operator==(const const_iterator& other) const { return pos_ == other.pos_; }
    bool operator!=(const const_iterator& other) const { return pos_ != other.pos_; }
    bool operator<(const const_iterator& other) const { return pos_ < other.pos_; }
    bool operator>(const const_iterator& other) const { return pos_ > other.pos_; }
    bool operator<=(const const_iterator& other) const { return pos_ <= other.pos_; }
    bool operator>=(const const_iterator& other) const { return pos_ >= other.pos_; }

   private:
    const StaticCircularBuffer* buf_ = nullptr;
    std::size_t pos_ = 0;
  };

  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, count_); }

  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, count_); }

  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

 private:
  std::array<T, N> buffer_;
  std::size_t start_;
  std::size_t count_;

  T& operator[](std::size_t i);
  const T& operator[](std::size_t i) const;
};

}  // namespace util

#include "inline/util/StaticCircularBuffer.inl"
