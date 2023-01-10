#include <connect4/C4PerfectPlayer.hpp>
#include <connect4/C4Tensorizor.hpp>

namespace c4 {

PerfectPlayParams PerfectPlayParams::global_params_;
Tensorizor::IdentityTransform Tensorizor::identity_transform_;
Tensorizor::ReflectionTransform Tensorizor::reflection_transform_;

}  // namespace c4
