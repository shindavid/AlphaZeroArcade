#pragma once

#include <vector>

namespace mit {

// Helper class for scheduler
class id_provider {
 public:
  int get_next_id();
  void recycle(int id);
  void clear();

 private:
  std::vector<int> recycled_ids_;
  int next_ = 0;
};

}  // namespace mit

#include <inline/util/mit/id_provider.inl>
