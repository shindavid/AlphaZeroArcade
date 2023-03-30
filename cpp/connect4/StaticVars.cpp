#include <connect4/C4PlayerFactory.hpp>
#include <connect4/C4Tensorizor.hpp>

namespace c4 {

PlayerFactory* PlayerFactory::instance_ = nullptr;
Tensorizor::IdentityTransform Tensorizor::identity_transform_;
Tensorizor::ReflectionTransform Tensorizor::reflection_transform_;

}  // namespace c4
