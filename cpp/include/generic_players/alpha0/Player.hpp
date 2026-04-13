#pragma once

// Redirect: generic::alpha0::Player<Spec> → alpha0::Player<Spec>

#include "alpha0/Player.hpp"

namespace generic::alpha0 {

template <typename Spec>
using Player = ::alpha0::Player<Spec>;

}  // namespace generic::alpha0
