#pragma once

#include "beta0/Calculations.hpp"
#include "search/LookupTable.hpp"
#include "search/SearchContext.hpp"
#include "search/TraitsTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/CompactBitSet.hpp"
#include "util/EigenUtil.hpp"

#include <sstream>

namespace beta0 {

template <search::concepts::Traits Traits>
class Backpropagator {
 public:
  using Edge = Traits::Edge;
  using Game = Traits::Game;
  using NodeStats = Traits::NodeStats;

  using LocalActionValueArray = Game::Types::LocalActionValueArray;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using ValueArray = Game::Types::ValueArray;

  using LookupTable = search::LookupTable<Traits>;
  using SearchContext = search::SearchContext<Traits>;
  using TraitsTypes = search::TraitsTypes<Traits>;

  using Node = TraitsTypes::Node;
  using Calculations = beta0::Calculations<Game>;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;

  using LocalArray = LocalPolicyArray;  // Alias for clarity
  using fresh_index_set_t = util::CompactBitSet<kMaxBranchingFactor>;

  template <typename MutexProtectedFunc>
  Backpropagator(SearchContext& context, Node* node, Edge* edge, MutexProtectedFunc&& func);

  ~Backpropagator();

 private:
  enum read_col_t : uint8_t {
    // Corresponds to columns of read_data_
    r_E,
    r_child_N,
    r_edge_N,
    r_R,
    r_P,
    r_pi,
    r_A,
    r_A_neg_inf,
    r_AV,
    r_lV,
    r_lU,
    r_lQ,
    r_lW,
    r_prev_lQ,
    r_prev_lW,

    r_Q,
    r_W,
    rSize
  };

  enum full_write_col_t : uint8_t {
    // Corresponds to columns of full_write_data_
    fw_lQ,
    fw_Q,
    fw_pi,
    fw_A,
    fw_A_neg_inf,
    fwSize
  };

  using Mask = eigen_util::DArray<kMaxBranchingFactor, bool>;
  using ReadArray = Eigen::Array<float, Eigen::Dynamic, rSize, 0, kMaxBranchingFactor>;
  using FullWriteArray = Eigen::Array<float, Eigen::Dynamic, fwSize, 0, kMaxBranchingFactor>;

  struct ReadData {
    void resize(int n) {
      array_.resize(n, rSize);
      array_.setZero();
    }

    auto operator()(read_col_t c) { return array_.col(c); }
    float& operator()(read_col_t c, int k) { return array_(k, c); }

    ReadArray array_;
  };

  struct FullWriteData {
    void resize(int n) {
      array_.resize(n, fwSize);
      array_.setZero();
    }

    auto operator()(full_write_col_t c) { return array_.col(c); }
    float& operator()(full_write_col_t c, int k) { return array_(k, c); }

    FullWriteArray array_;
  };

  bool shares_mutex_with_parent(const Node* child) const;
  void load_child_stats(int i, const NodeStats& child_stats);

  void preload_parent_data();
  template <typename MutexProtectedFunc>
  void load_parent_data(MutexProtectedFunc&& func);
  void load_remaining_data();
  void compute_update_rules();
  void apply_updates();
  void print_debug_info();

  bool handle_edge_cases();  // return true if can short-circuit
  void update_Q_estimates();
  void compute_ratings();
  bool compute_ratings_helper(int i);  // return true if can short-circuit
  void compute_policy();
  LocalArray compute_tau(float lQ_i, const LocalArray& lQ, float lW_i, const LocalArray& lW,
                         const LocalArray& z, const LocalArray& lU_rsqrt);
  void update_R();
  void update_QW();
  void safety_check(int line);
  std::ostringstream& debug_ss();
  void debug_flush();
  void fail(const std::string& message);

  LocalArray splice(const LocalArray& x, int i);
  LocalArray unsplice(const LocalArray& x, int i);
  LookupTable& lookup_table() { return context_.general_context->lookup_table; }

  SearchContext& context_;
  NodeStats stats_;
  Node* node_;
  fresh_index_set_t fresh_indices_;
  Mask E_mask_;
  Mask U_mask_;
  int n_;  // number of valid actions
  float Q_floor_;
  core::seat_index_t seat_;

  int num_deferred_child_stats_load_indices_ = 0;
  int deferred_child_stats_load_indices_[Game::Constants::kMaxBranchingFactor];

  ReadData read_data_;
  FullWriteData full_write_data_;
  std::ostringstream* debug_ss_ = nullptr;
  bool debug_info_printed_ = false;
};

}  // namespace beta0

#include "inline/beta0/Backpropagator.inl"
