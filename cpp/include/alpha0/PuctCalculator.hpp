#pragma once

#include "alpha0/Edge.hpp"
#include "alpha0/ManagerParams.hpp"
#include "alpha0/Node.hpp"
#include "core/BasicTypes.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchParams.hpp"
#include "alpha0/concepts/SpecConcept.hpp"

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
struct PuctCalculator {
  using Game = Spec::Game;
  using Edge = alpha0::Edge<Spec>;
  using LookupTable = search::LookupTable<Spec>;
  using ManagerParams = alpha0::ManagerParams<Spec>;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;

  using Node = alpha0::Node<Spec>;

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

}  // namespace alpha0

#include "inline/alpha0/PuctCalculator.inl"
