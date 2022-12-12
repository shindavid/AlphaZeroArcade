#pragma once

#include <array>
#include <bit>
#include <bitset>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

/*
 * TODO: incorporate this into <util/BitSet.hpp>
 */
template<size_t N>
struct SetBitIndexSet {
  using bitset_t = std::bitset<N>;
  static constexpr size_t kNumPages = sizeof(bitset_t) / 8;

  struct Iterator {
    Iterator() : pages_(nullptr), cur_page_(0), page_num_(kNumPages) {}

    Iterator(const SetBitIndexSet* set, int page_num)
        : pages_(set->pages_)
          , cur_page_(pages_[page_num])
          , page_num_(page_num)
    {
      skip_to_next();
    }

    bool operator==(const Iterator& other) const {
      return cur_page_ == other.cur_page_ && page_num_ == other.page_num_;
    }

    bool operator!=(const Iterator& other) const {
      return !(*this == other);
    }

    size_t operator*() const { return 64 * page_num_ + get_index(); }

    Iterator& operator++() {
      cur_page_ -= (1UL << get_index());
      skip_to_next();
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

  private:
    int get_index() const {
      return std::countr_zero(cur_page_);
    }

    void skip_to_next() {
      if (cur_page_) return;

      while (page_num_ < kNumPages && !cur_page_) {
        cur_page_ = pages_[++page_num_];
      }
      cur_page_ *= (page_num_ != kNumPages);
    }

    const uint64_t* const pages_;
    uint64_t cur_page_;
    size_t page_num_ = 0;
  };

  SetBitIndexSet(const bitset_t& bitset)
      : pages_(reinterpret_cast<const uint64_t*>(&bitset))
  {}

  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(); }

  const uint64_t* const pages_;
};

template<size_t N>
struct SinglePageSetBitIndexSet {
  using bitset_t = std::bitset<N>;
  static_assert(sizeof(bitset_t) == 8);

  struct Iterator {
    Iterator() : cur_value_(0) {}

    Iterator(const SinglePageSetBitIndexSet* set)
        : cur_value_(set->value_) {}

    bool operator==(const Iterator& other) const {
      return cur_value_ == other.cur_value_;
    }

    bool operator!=(const Iterator& other) const {
      return !(*this == other);
    }

    size_t operator*() const { return get_index(); }

    Iterator& operator++() {
      cur_value_ -= (1UL << get_index());
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

  private:
    int get_index() const {
      return std::countr_zero(cur_value_);
    }

    uint64_t cur_value_;
  };

  SinglePageSetBitIndexSet(const bitset_t& bitset)
      : value_(reinterpret_cast<const uint64_t&>(bitset))
  {}

  Iterator begin() const { return Iterator(this); }
  Iterator end() const { return Iterator(); }

  const uint64_t value_;
};

template<size_t N>
typename std::enable_if_t<(N > 64), SetBitIndexSet<N>> get_set_bits(const std::bitset<N>& bitset) {
  return SetBitIndexSet<N>(bitset);
}

template<size_t N>
typename std::enable_if_t<(N <= 64), SinglePageSetBitIndexSet<N>> get_set_bits(const std::bitset<N>& bitset) {
  return SinglePageSetBitIndexSet<N>(bitset);
}

void dummy(int*, int*);

template<int N>
void do_subexperiment() {
  for (int K=0; K<N; ++K) {
    std::bitset<N> bitset;
    std::array<int, N> indices;
    std::vector<int> selected_indices;
    for (int i=0; i<N; ++i) {
      indices[i] = i;
    }
    std::sample(indices.begin(), indices.end(), std::back_inserter(selected_indices), K,
                std::mt19937{std::random_device{}()});
    for (int i : selected_indices) {
      bitset[i] = 1;
    }

    std::array<int, N> set_indices;
    std::array<int, N> set_indices2;
    int s;

    constexpr int kNumIterations = 100;
    std::array<int64_t, kNumIterations> clockings1, clockings2;

    for (int e = 0; e < kNumIterations; ++e) {
      auto t1 = std::chrono::high_resolution_clock::now();
      s = 0;
      for (int i = 0; i < N; ++i) {
        if (bitset[i]) {
          set_indices[s++] = i;
        }
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      s = 0;
      for (int i: get_set_bits(bitset)) {
        set_indices2[s++] = i;
      }
      auto t3 = std::chrono::high_resolution_clock::now();
      dummy(set_indices.data(), set_indices2.data());

      auto t12 = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
      auto t23 = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2);

      clockings1[e] = t12.count();
      clockings2[e] = t23.count();
    }

    std::sort(clockings1.begin(), clockings1.end());
    std::sort(clockings2.begin(), clockings2.end());

    int64_t u1 = 0;
    int64_t u2 = 0;
    constexpr int kNumOutliers = 5;
    for (int u = kNumOutliers; u < kNumIterations - kNumOutliers; ++u) {
      u1 += clockings1[u];
      u2 += clockings2[u];
    }
    u1 /= (kNumIterations - 2 * kNumOutliers);
    u2 /= (kNumIterations - 2 * kNumOutliers);
    std::cout << "N=" << N << " K=" << K << " u1=" << u1 << "ns u2=" << u2 << "ns" << std::endl;
  }
}

template<typename T, T... Ints>
void do_experiment(std::integer_sequence<T, Ints...> seq) {
  ((do_subexperiment<Ints+1>()), ...);
}
