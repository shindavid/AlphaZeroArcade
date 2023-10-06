#pragma once

#include <array>
#include <cstdint>
#include <functional>

#include <boost/functional/hash.hpp>
#include <torch/torch.h>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/SerializerTypes.hpp>
#include <core/serializers/DeterministicGameSerializer.hpp>
#include <games/blokus/Constants.hpp>
#include <mcts/SearchResults.hpp>
#include <mcts/SearchResultsDumper.hpp>
#include <util/EigenUtil.hpp>

namespace blokus { class GameState; }

template <>
struct std::hash<blokus::GameState> {
  std::size_t operator()(const blokus::GameState& state) const;
};

namespace blokus {

class GameState {};

static_assert(core::GameStateConcept<blokus::GameState>);

using Player = core::AbstractPlayer<GameState>;

}  // namespace blokus

namespace core {

// template specialization
template<> struct serializer<blokus::GameState> {
  using type = DeterministicGameSerializer<blokus::GameState>;
};

}  // namespace core

namespace mcts {

template<> struct SearchResultsDumper<blokus::GameState> {
  using LocalPolicyArray = blokus::GameState::LocalPolicyArray;
  using SearchResults = mcts::SearchResults<blokus::GameState>;

  static void dump(const LocalPolicyArray& action_policy, const SearchResults& results);
};

}  // namespace mcts

#include <games/blokus/inl/GameState.inl>
