#pragma once

#include <util/EigenUtil.hpp>

namespace core {

template<eigen_util::FTensor PolicyTensor>
struct PolicyTarget {};

template <eigen_util::FArray ValueArray>
struct ValueTarget {};

template <eigen_util::FTensor PolicyTensor>
struct OppPolicyTarget {};

}  // namespace core
