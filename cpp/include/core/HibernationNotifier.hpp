#pragma once

#include <core/BasicTypes.hpp>
#include <core/HibernationManager.hpp>

namespace core {

class HibernationNotifier {
 public:
  HibernationNotifier(HibernationManager* manager, game_slot_index_t slot_id)
      : manager_(manager), slot_id_(slot_id) {}

  void notify() { manager_->notify(slot_id_); }

 private:
  HibernationManager* const manager_;
  const game_slot_index_t slot_id_;
};

}  // namespace core
