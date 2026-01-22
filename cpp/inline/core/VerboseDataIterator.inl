#include "core/VerboseDataIterator.hpp"

namespace core {

template <concepts::Game Game>
typename VerboseDataIterator<Game>::VerboseData_sptr VerboseDataIterator<Game>::most_recent_data() const {
  auto ix = index_;
  while (ix >= 0) {
    const auto& data = tree_->verbose_data(ix);
    if (data) {
      return data;
    }
    ix = tree_->get_parent_index(ix);
  }
  return nullptr;
}

} // namespace core
