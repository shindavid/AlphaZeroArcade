#pragma once

#include "search/LookupTable.hpp"
#include "search/SearchContext.hpp"
#include "search/TraitsTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/Gaussian1D.hpp"

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

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;

  using LocalArray = LocalPolicyArray;  // Alias for clarity

  template <typename MutexProtectedFunc>
  Backpropagator(SearchContext& context, Node* node, Edge* edge, MutexProtectedFunc&& func);

 private:
  enum read_col_t : uint8_t {
    // Corresponds to columns of read_data_
    r_P,
    r_pi,
    r_A,
    r_lV,
    r_lU,
    r_lQ,
    r_lW,
    rSize
  };

  enum read_col_2d_t : uint8_t {
    // Corresponds to column-chunks of read_data2_
    r2_Q,
    r2_W,
    r2_Q_star,
    rSize2
  };

  enum full_write_col_t : uint8_t {
    // Corresponds to columns of full_write_data_
    fw_pi,
    fw_A,
    fwSize
  };

  enum sibling_read_col_t : uint8_t {
    // Corresponds to columns of sibling_read_data_
    sr_P,
    sr_pi,
    sr_A,
    sr_lV,
    sr_lU,
    sr_lQ,
    sr_lW,
    srSize
  };

  enum sibling_write_col_t : uint8_t {
    // Corresponds to columns of sibling_write_data_
    sw_c,
    sw_z,
    sw_w,
    sw_tau,
    swSize
  };

  using ReadArray = Eigen::Array<float, Eigen::Dynamic, rSize, 0, kMaxBranchingFactor>;
  using ReadArray2D =
    Eigen::Array<float, Eigen::Dynamic, rSize2 * kNumPlayers, 0, kMaxBranchingFactor>;
  using FullWriteArray = Eigen::Array<float, Eigen::Dynamic, fwSize, 0, kMaxBranchingFactor>;
  using SiblingReadArray = Eigen::Array<float, Eigen::Dynamic, srSize, 0, kMaxBranchingFactor>;
  using SiblingWriteArray = Eigen::Array<float, Eigen::Dynamic, swSize, 0, kMaxBranchingFactor>;

  struct ReadData {
    void resize(int n) { array_.resize(n, rSize); zero_out_in_debug_mode(array_); }

    auto operator()(read_col_t c) { return array_.col(c); }
    float& operator()(read_col_t c, int k) { return array_(k, c); }

    ReadArray array_;
  };

  struct ReadData2D {
    static constexpr int P = kNumPlayers;
    void resize(int n) { array_.resize(n, P * rSize2); zero_out_in_debug_mode(array_); }

    // Returns a (N, P) shaped block
    auto operator()(read_col_2d_t c) { return array_.middleCols(P * c, P); }

    // Returns a (1, P) shaped block
    auto operator()(read_col_2d_t c, int k) { return array_.block(k, P * c, 1, P); }

    float& operator()(read_col_2d_t c, int k, core::seat_index_t s) { return array_(k, P * c + s); }

    ReadArray2D array_;
  };

  struct FullWriteData {
    void resize(int n) { array_.resize(n, fwSize); zero_out_in_debug_mode(array_); }

    auto operator()(full_write_col_t c) { return array_.col(c); }
    float& operator()(full_write_col_t c, int k) { return array_(k, c); }

    FullWriteArray array_;
  };

  struct SiblingReadData {
    void resize(int n) { array_.resize(n, srSize); zero_out_in_debug_mode(array_); }

    auto operator()(sibling_read_col_t c) { return array_.col(c); }
    float& operator()(sibling_read_col_t c, int k) { return array_(k, c); }

    SiblingReadArray array_;
  };

  struct SiblingWriteData {
    void resize(int n) { array_.resize(n, swSize); zero_out_in_debug_mode(array_); }

    auto operator()(sibling_write_col_t c) { return array_.col(c); }
    float& operator()(sibling_write_col_t c, int k) { return array_(k, c); }

    SiblingWriteArray array_;
  };

  static void zero_out_in_debug_mode(auto& array) {
    if (IS_DEFINED(DEBUG_BUILD) || search::kEnableSearchDebug) {
      array.setZero();
    }
  }

  bool shares_mutex_with_parent(const Node* child) const;
  void load_child_stats(int i, const NodeStats& child_stats);

  void preload_parent_data();
  template <typename MutexProtectedFunc>
  void load_parent_data(MutexProtectedFunc&& func);
  void load_remaining_data();
  void compute_update_rules();
  void apply_updates();
  void print_debug_info();

  void compute_ratings();
  void calibrate_ratings();
  void compute_policy();
  LocalArray compute_tau(float lQ_i, const LocalArray& lQ, float lW_i, const LocalArray& lW,
                         const LocalArray& z);
  void solve_for_A_i(const LocalArray& w, const LocalArray& tau, const LocalArray& A, float& A_i);
  void update_QW();

  void splice(read_col_t from_col, sibling_read_col_t to_col);
  LocalArray unsplice(sibling_write_col_t from_col);

  // Sets Q_arr(action_index, seat) = q_new, and adjusts other players' Q values accordingly.
  template <typename T>
  static void modify_Q_arr(T& Q_arr, int action_index, core::seat_index_t seat, float q_new);

  template <typename T>
  void normalize_policy(T pi);  // keep pi[i_] fixed, normalize others

  LookupTable& lookup_table() { return context_.general_context->lookup_table; }

  SearchContext& context_;
  NodeStats stats_;
  Node* node_;
  Edge* edge_;
  int n_;  // number of valid actions
  int i_;  // current action index
  util::Gaussian1D lQW_i_old_;
  float Q_floor_;
  core::seat_index_t seat_;

  const Node* child_i_ = nullptr;  // useful for debugging
  int num_deferred_child_stats_load_indices_ = 0;
  int deferred_child_stats_load_indices_[Game::Constants::kMaxBranchingFactor];

  ReadData read_data_;
  ReadData2D read_data2_;
  FullWriteData full_write_data_;
  SiblingReadData sibling_read_data_;
  SiblingWriteData sibling_write_data_;
};

}  // namespace beta0

#include "inline/beta0/Backpropagator.inl"
