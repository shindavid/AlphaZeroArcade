#include <core/OraclePool.hpp>

#include <util/Asserts.hpp>

namespace core {

template <class OracleT>
OraclePool<OracleT>::~OraclePool() {
  for (OracleT* oracle : free_oracles_) {
    delete oracle;
  }
}

template <class OracleT>
void OraclePool<OracleT>::set_capacity(size_t capacity) {
  util::release_assert(all_oracles_.empty(), "Cannot change capacity after oracles are created");
  capacity_ = capacity;
  all_oracles_.reserve(capacity);
}

template <class OracleT>
template <typename... Ts>
OracleT* OraclePool<OracleT>::get_oracle(const YieldNotificationUnit& unit,
                                         Ts&&... constructor_args) {
  std::unique_lock lock(mutex_);
  if (!free_oracles_.empty()) {
    OracleT* oracle = free_oracles_.back();
    free_oracles_.pop_back();
    return oracle;
  }
  if (all_oracles_.size() < capacity_) {
    OracleT* oracle = new OracleT(std::forward<Ts>(constructor_args)...);
    all_oracles_.push_back(oracle);
    return oracle;
  }

  if (unit.yield_manager) {
    pending_notification_units_.push_back(unit);
  }
  return nullptr;
}

template <class OracleT>
void OraclePool<OracleT>::release_oracle(OracleT* oracle) {
  std::unique_lock lock(mutex_);
  free_oracles_.push_back(oracle);
  if (!pending_notification_units_.empty()) {
    YieldNotificationUnit unit = pending_notification_units_.back();
    pending_notification_units_.pop_back();
    SlotContext slot_context(unit.game_slot_index, unit.context_id);
    unit.yield_manager->notify(slot_context);
  }
}

}  // namespace core
