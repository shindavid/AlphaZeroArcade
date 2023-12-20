#include <core/PerfStats.hpp>

namespace core {

inline perf_stats_t& perf_stats_t::operator+=(const perf_stats_t& other) {
  cache_hits += other.cache_hits;
  cache_misses += other.cache_misses;
  positions_evaluated += other.positions_evaluated;
  batches_evaluated += other.batches_evaluated;
  full_batches_evaluated += other.full_batches_evaluated;
  return *this;
}

inline boost::json::object perf_stats_t::to_json() const {
  boost::json::object obj;
  obj["cache_hits"] = cache_hits;
  obj["cache_misses"] = cache_misses;
  obj["positions_evaluated"] = positions_evaluated;
  obj["batches_evaluated"] = batches_evaluated;
  obj["full_batches_evaluated"] = full_batches_evaluated;
  return obj;
}

}  // namespace core
