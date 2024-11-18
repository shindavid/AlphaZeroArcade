#include <util/InfiniteBitset.hpp>

namespace util {

inline InfiniteBitset::reference InfiniteBitset::operator[](size_t pos) {
  auto_resize(pos);
  return bits_[pos];
}

inline void InfiniteBitset::set(size_t n) {
  auto_resize(n);
  bits_.set(n);
}

inline void InfiniteBitset::reset() { bits_.reset(); }

inline size_t InfiniteBitset::count() const { return bits_.count(); }

inline void InfiniteBitset::auto_resize(size_t pos) {
  if (pos >= bits_.size()) {
    bits_.resize(pos * 2);
  }
}

}  // namespace util
