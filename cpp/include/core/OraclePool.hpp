#pragma once

#include <core/BasicTypes.hpp>
#include <core/YieldManager.hpp>
#include <util/mit/mit.hpp>

namespace core {

/*
 * For certain games, we have access to a third-party binary that can be queried like an oracle to
 * evaluate game states. This class manages a pool of oracles, and provides mechanics compatible
 * with the GameServer class.
 */
template <class OracleT>
class OraclePool {
 public:
  // capacity = max number of oracles instances to create
  OraclePool(size_t capacity = 16) { set_capacity(capacity); }
  ~OraclePool();

  size_t capacity() const { return capacity_; }
  void set_capacity(size_t capacity);

  // If there are fewer than capacity_ busy oracles, then returns a free oracle (creating a new one
  // if necessary). Otherwise, returns nullptr, and schedules unit to be notified when an oracle
  // becomes free.
  //
  // If a new oracle needs to be created, then the constructor of OracleT is called with the
  // passed-in arguments.
  template<typename... Ts>
  OracleT* get_oracle(const YieldNotificationUnit& unit, Ts&&... constructor_args);

  void release_oracle(OracleT* oracle);

 private:
  using oracle_vec_t = std::vector<OracleT*>;
  using notification_unit_vec_t = std::vector<YieldNotificationUnit>;

  oracle_vec_t free_oracles_;
  oracle_vec_t all_oracles_;
  notification_unit_vec_t pending_notification_units_;
  mutable mit::mutex mutex_;
  size_t capacity_;
};

}  // namespace core

#include <inline/core/OraclePool.inl>
