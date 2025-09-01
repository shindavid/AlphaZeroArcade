#pragma once

namespace search {
namespace concepts {

template <class A, class ManagerParams>
concept AuxState = requires(A& a, const ManagerParams& mparams) {
  { A(mparams) };
  { a.clear() };
  { a.step() };
};

}  // namespace concepts
}  // namespace search
