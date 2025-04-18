#pragma once

#include <core/BasicTypes.hpp>
#include <core/HibernationNotifier.hpp>

#include <mutex>
#include <type_traits>

namespace core {

/*
 * For certain games, we have access to a third-party binary that can be queried like an oracle to
 * evaluate game states. This class manages a pool of oracles, and provides mechanics compatible
 * with the GameServer class.
 */
template <class OracleT>
class OraclePool {
 public:
  static_assert(std::is_default_constructible_v<OracleT>);

  // capacity = max number of oracles instances to create
  OraclePool(size_t capacity = 16) { set_capacity(capacity); }
  ~OraclePool();

  void set_capacity(size_t capacity);

  // If there are fewer than capacity_ busy oracles, then returns a free oracle (creating a new one
  // if necessary). Otherwise, returns nullptr, and schedules the passed-in notifier to be
  // notified when an oracle becomes available.
  OracleT* get_oracle(core::HibernationNotifier* notifier = nullptr);

  void release_oracle(OracleT* oracle);

 private:
  using oracle_vec_t = std::vector<OracleT*>;
  using notifier_vec_t = std::vector<core::HibernationNotifier*>;

  oracle_vec_t free_oracles_;
  oracle_vec_t all_oracles_;
  notifier_vec_t pending_notifiers_;
  mutable std::mutex mutex_;
  size_t capacity_;
};

}  // namespace core

#include <inline/core/OraclePool.inl>
