#pragma once

#include "search/LookupTable.hpp"
#include "search/SearchContext.hpp"
#include "search/TraitsTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace beta0 {

template <search::concepts::Traits Traits>
class Backpropagator {
 public:
  using Edge = Traits::Edge;
  using Game = Traits::Game;
  using NodeStats = Traits::NodeStats;

  using LocalActionValueArray = Game::Types::LocalActionValueArray;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;

  using LookupTable = search::LookupTable<Traits>;
  using SearchContext = search::SearchContext<Traits>;
  using TraitsTypes = search::TraitsTypes<Traits>;

  using Node = TraitsTypes::Node;

  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;
  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;

  // // Aliases for clarity
  // using LocalArray1D = LocalPolicyArray;
  // using LocalArray2D = LocalActionValueArray;

  Backpropagator(SearchContext& context, Node* node, Edge* edge);

 private:
  enum read_col_t : uint8_t {
    // Corresponds to columns of read_data_
    r_P,
    r_pi,
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
    fw_pi_minus,  // TODO: we only need this for i_
    fw_pi_plus,   // TODO: we only need this for i_
    fw_Qp_minus,
    fw_Qp_plus,
    fwSize
  };

  enum sibling_read_col_t : uint8_t {
    // Corresponds to columns of sibling_read_data_
    sr_P,
    sr_pi,
    sr_lV,
    sr_lU,
    sr_lQ,
    sr_lW,
    sr_Qp_minus,
    sr_Qp_plus,
    sr_pi_minus,
    sr_pi_plus,
    srSize
  };

  enum sibling_write_col_t : uint8_t {
    // Corresponds to columns of sibling_write_data_
    sw_S_denom_inv,
    sw_S,
    sw_c,
    sw_z,
    sw_tau,
    sw_S_minus,
    sw_S_plus,
    sw_tau_minus,
    sw_tau_plus,
    sw_Qp_minus,
    sw_Qp_plus,
    swSize
  };

  using ReadArray = Eigen::Array<float, Eigen::Dynamic, rSize, 0, kMaxBranchingFactor>;
  using ReadArray2D =
    Eigen::Array<float, Eigen::Dynamic, rSize2 * kNumPlayers, 0, kMaxBranchingFactor>;
  using FullWriteArray = Eigen::Array<float, Eigen::Dynamic, fwSize, 0, kMaxBranchingFactor>;
  using SiblingReadArray = Eigen::Array<float, Eigen::Dynamic, srSize, 0, kMaxBranchingFactor>;
  using SiblingWriteArray = Eigen::Array<float, Eigen::Dynamic, swSize, 0, kMaxBranchingFactor>;

  struct ReadData {
    void resize(int n) { array_.resize(rSize, n); }

    auto operator()(read_col_t c) { return array_.col(c); }
    float& operator()(read_col_t c, int k) { return array_(k, c); }

    ReadArray array_;
  };

  struct ReadData2D {
    static constexpr int P = kNumPlayers;
    void resize(int n) { array_.resize(P * rSize2, n); }

    // Returns a (N, P) shaped block
    auto operator()(read_col_2d_t c) { return array_.middleCols(P * c, P); }

    // Returns a (1, P) shaped block
    auto operator()(read_col_2d_t c, int k) { return array_.block(k, P * c, 1, P); }

    float& operator()(read_col_2d_t c, int k, core::seat_index_t s) { return array_(k, P * c + s); }

    ReadArray2D array_;
  };

  struct FullWriteData {
    void resize(int n) { array_.resize(fwSize, n); }

    auto operator()(full_write_col_t c) { return array_.col(c); }
    float& operator()(full_write_col_t c, int k) { return array_(k, c); }

    FullWriteArray array_;
  };

  struct SiblingReadData {
    void resize(int n) { array_.resize(srSize, n); }

    auto operator()(sibling_read_col_t c) { return array_.col(c); }
    float& operator()(sibling_read_col_t c, int k) { return array_(k, c); }

    SiblingReadArray array_;
  };

  struct SiblingWriteData {
    void resize(int n) { array_.resize(swSize, n); }

    auto operator()(sibling_write_col_t c) { return array_.col(c); }
    float& operator()(sibling_write_col_t c, int k) { return array_(k, c); }

    SiblingWriteArray array_;
  };

  bool shares_mutex_with_parent(const Node* child) const;
  void load_child_stats(int i, const NodeStats& child_stats);

  void preload_parent_data();
  void load_parent_data();
  void load_remaining_data();
  void compute_update_rules();
  void apply_updates();

  void compute_policy();
  // void compute_Q_stars();
  void update_QW();
  void update_Qp_plus_minus();

  void splice(read_col_t from_col, sibling_read_col_t to_col);
  void unsplice(sibling_write_col_t from_col, full_write_col_t to_col);

  // Sets Q_arr(action_index, seat) = q_new, and adjusts other players' Q values accordingly.
  template<typename T>
  static void modify_Q_arr(T& Q_arr, int action_index,
                           core::seat_index_t seat, float q_new);

  template <typename T>
  void normalize_policy(T pi);

  LookupTable& lookup_table() { return context_.general_context->lookup_table; }

  SearchContext& context_;
  NodeStats stats_;
  Node* node_;
  Edge* edge_;
  int n_;  // number of valid actions
  int i_;  // current action index
  float Qi_snapshot_;
  float Q_floor_;
  core::seat_index_t seat_;

  int num_deferred_child_stats_load_indices_ = 0;
  int deferred_child_stats_load_indices_[Game::Constants::kMaxBranchingFactor];

  ReadData read_data_;
  ReadData2D read_data2_;
  FullWriteData full_write_data_;
  SiblingReadData sibling_read_data_;
  SiblingWriteData sibling_write_data_;

  // struct WriteData {
  //   struct FullData {
  //     // full data arrays are of length n_
  //   };

  //   struct SiblingData {
  //     // sibling data arrays are of length (n_ - 1), as they splice out column i_
  //   };
  // };

  // // full data arrays are of length n_
  // struct FullData {
  //   void resize(int n);

  //   LocalArray1D P;
  //   LocalArray1D pi;
  //   LocalArray1D lV;
  //   LocalArray1D lU;
  //   LocalArray1D lQ;
  //   LocalArray1D lW;
  //   LocalArray1D Q_star_minus;
  //   LocalArray1D Q_star_plus;

  //   LocalArray2D Q;
  //   LocalArray2D W;
  //   LocalArray2D Q_star;
  // };

  // // sibling data arrays are of length (n_ - 1), as they split out column i_
  // struct SiblingData {
  //   void resize(int n);

  //   LocalArray1D P;
  //   LocalArray1D pi;
  //   LocalArray1D lV;
  //   LocalArray1D lU;
  //   LocalArray1D lQ;
  //   LocalArray1D lW;

  //   LocalArray1D S_denom_inv;
  //   LocalArray1D S_minus;
  //   LocalArray1D S;
  //   LocalArray1D S_plus;
  //   LocalArray1D c;
  //   LocalArray1D z;
  //   LocalArray1D tau_minus;
  //   LocalArray1D tau;
  //   LocalArray1D tau_plus;

  //   LocalArray1D Qp_minus;
  //   LocalArray1D Qp_plus;
  // };

  // struct UpdateData {
  //   void resize(int n);

  //   LocalArray1D pi_minus;  // posterior policy if we shock Q[i] downward
  //   LocalArray1D pi;        // posterior policy given actual Q[i]
  //   LocalArray1D pi_plus;   // posterior policy if we shock Q[i] upward

  //   LocalArray1D Qp_minus;
  //   LocalArray1D Qp_plus;
  // };

  // FullData full_data_;
  // SiblingData sibling_data_;
  // UpdateData update_data_;
};

}  // namespace beta0

#include "inline/betazero/Backpropagator.inl"
