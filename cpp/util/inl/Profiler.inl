#include <util/Profiler.hpp>

#include <util/CppUtil.hpp>

namespace util {

template<int NumRegions, bool Verbose>
void Profiler<NumRegions, Verbose>::record(int region) {
  time_point_t now = clock_t::now();
  if (Verbose) {
    int64_t ns = util::ns_since_epoch(now);
    printf("%lu.%09lu %s %d\n", ns / 1000000000, ns % 1000000000, name_.c_str(), (int) region);
  }

  if (cur_region_ != kNumRegions) {
    std::chrono::nanoseconds duration = now - last_time_;
    durations_[cur_region_] += duration;
  }
  last_time_ = now;
  cur_region_ = region;
}

template<int NumRegions, bool Verbose>
void Profiler<NumRegions, Verbose>::clear() {
  for (int r = 0; r < kNumRegions; ++r) {
    durations_[r] *= 0;
  }
  count_ = 0;
  cur_region_ = kNumRegions;
}

template<int NumRegions, bool Verbose>
void Profiler<NumRegions, Verbose>::dump(int count) {
  if (--skip_count_ >= 0) return clear();
  if (++count_ < count) return;
  fprintf(file_, "%s dump n=%d\n", name_.c_str(), count_);
  double inv_count = 1.0 / count_;
  int64_t ns_total = 0;
  for (int r = 0; r < kNumRegions; ++r) {
    int64_t ns = durations_[r].count();
    ns_total += ns;
  }
  double inv_ns_total = ns_total ? 1.0 / ns_total : 1;
  for (int r = 0; r < kNumRegions; ++r) {
    int64_t ns = durations_[r].count();
    if (!ns) continue;
    double avg_ns = ns * inv_count;
    double pct = 100.0 * ns * inv_ns_total;
    fprintf(file_, "%2d %9.f %7.2f%%\n", r, avg_ns, pct);
  }
  clear();
}

}  // namespace util
