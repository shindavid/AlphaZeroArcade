#include <core/Symmetries.hpp>

namespace core {

template <concepts::Game Game>
Transforms<Game>* Transforms<Game>::instance_ = nullptr;

template <concepts::Game Game>
Transforms<Game>::Transforms() {
  mp::constexpr_for<0, kNumTransforms, 1>([&](auto i) {
    using T = mp::TypeAt_t<TransformList, i>;
    transforms_[i] = new T();
  });
}

template <concepts::Game Game>
Transforms<Game>* Transforms<Game>::instance() {
  if (!instance_) {
    instance_ = new Transforms();
  }
  return instance_;
}

template <concepts::Game Game>
typename Transforms<Game>::Transform* Transforms<Game>::get(core::symmetry_index_t index) {
  return instance()->transforms_[index];
}

}  // namespace core
