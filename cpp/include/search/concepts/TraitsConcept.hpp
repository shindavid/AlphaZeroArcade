#pragma once

#include "core/concepts/Game.hpp"

namespace search {
namespace concepts {

template <class T>
concept Traits = requires {
  requires core::concepts::Game<typename T::Game>;
  typename T::Node;
  typename T::Edge;
  typename T::AuxState;
  typename T::ManagerParams;
  typename T::Algorithms;
  typename T::EvalRequest;
  typename T::EvalResponse;
  typename T::EvalServiceBase;
  typename T::EvalServiceFactory;
};

}  // namespace concepts
}  // namespace search
