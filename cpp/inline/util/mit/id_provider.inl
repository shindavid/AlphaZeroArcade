#include <util/mit/id_provider.hpp>

namespace mit {

inline int id_provider::get_next_id() {
  if (!recycled_ids_.empty()) {
    int id = recycled_ids_.back();
    recycled_ids_.pop_back();
    return id;
  }
  return next_++;
}

inline void id_provider::recycle(int id) { recycled_ids_.push_back(id); }

inline void id_provider::clear() {
  recycled_ids_.clear();
  next_ = 0;
}

}  // namespace mit
