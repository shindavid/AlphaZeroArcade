#pragma once

#include "beta0/Edge.hpp"
#include "beta0/GraphTraits.hpp"
#include "beta0/ManagerParams.hpp"
#include "beta0/Node.hpp"
#include "beta0/concepts/SpecConcept.hpp"
#include "core/BasicTypes.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchParams.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
struct PuctCalculator {
  using Game = Spec::Game;
  using Edge = beta0::Edge<Spec>;
  using LookupTable = search::LookupTable<beta0::GraphTraits<Spec>>;
  using ManagerParams = beta0::ManagerParams<Spec>;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;

  using Node = beta0::Node<Spec>;

  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;
  static constexpr float eps = 1e-6;  // needed when N == 0

  PuctCalculator(const LookupTable& lookup_table, const ManagerParams& manager_params,
                 const search::SearchParams& search_params, const Node* node, bool is_root);

  core::seat_index_t seat;
  LocalPolicyArray P;
  LocalPolicyArray Q;    // (virtualized) value
  LocalPolicyArray PW;   // provably-winning
  LocalPolicyArray PL;   // provably-losing
  LocalPolicyArray E;    // edge count
  LocalPolicyArray mE;   // masked edge count
  LocalPolicyArray RN;   // real node count
  LocalPolicyArray VN;   // virtual node count
  LocalPolicyArray FPU;  // FPU
  LocalPolicyArray PUCT;
};

}  // namespace beta0

#include "inline/beta0/PuctCalculator.inl"
