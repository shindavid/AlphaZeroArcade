#pragma once

#include <chrono>

namespace util {

/*
 * Keeps track of how much is spent in each of various execution regions.
 *
 * Usage:
 *
 * enum Region {
 *   kDoingFoo,
 *   kDoingBar,
 *   kNumRegions
 * };
 *
 * FILE* file = ...;
 * using profiler_t = util::Profiler<int(kNumRegions)>;
 * profiler_t profiler;
 * profiler.skip_next_n_dumps(5);  // to get rid of program startup noise
 * for (int i = 0; i < 64; ++i) {
 *   profiler.record(kDoingFoo);
 *   do_foo();
 *   profiler.record(kDoingBar);
 *   do_bar();
 *   profiler.dump(file, 10);  // dump every 10 times
 * }
 * profiler.dump(file);  // final dump
 */
template<int NumRegions, bool Verbose=false>
class Profiler {
public:
  using clock_t = std::chrono::steady_clock;
  using time_point_t = std::chrono::time_point<clock_t>;
  static constexpr int kNumRegions = NumRegions;
  static constexpr bool kVerbose = Verbose;

  Profiler() { clear(); }
  int count() const { return count_; }
  void skip_next_n_dumps(int n) { skip_count_ = n; }
  void record(int region, const char* name="");
  void clear();
  void dump(FILE* file, int count=1, const char* name="");  // dump only if ++count_ == count

private:
  void dump_helper(FILE* file, const char* name, int count);

  std::chrono::nanoseconds durations_[kNumRegions];
  time_point_t last_time_;
  int skip_count_ = 0;
  int count_ = 0;
  int cur_region_;
};

}  // namespace util

#include <util/inl/Profiler.inl>
