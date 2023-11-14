#include <mcts/TreeTraversalThread.hpp>

#include <util/Asserts.hpp>

#include <boost/algorithm/string.hpp>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline TreeTraversalThread<GameState, Tensorizor>::TreeTraversalThread(
    TreeTraversalMode traversal_mode, const boost::filesystem::path& profiling_dir, int thread_id)
    : traversal_mode_(traversal_mode), thread_id_(thread_id) {
  if (kEnableProfiling) {
    auto profiling_file_path = profiling_dir / util::create_string("search-%d.txt", thread_id);
    profiler_.initialize_file(profiling_file_path);
    profiler_.set_name(util::create_string("s-%-3d", thread_id));
    profiler_.skip_next_n_dumps(5);
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline TreeTraversalThread<GameState, Tensorizor>::~TreeTraversalThread() {
  kill();
  profiler_.dump(1);
  profiler_.close_file();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void TreeTraversalThread<GameState, Tensorizor>::join() {
  if (thread_ && thread_->joinable()) {
    thread_->join();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void TreeTraversalThread<GameState, Tensorizor>::kill() {
  join();
  if (thread_) {
    delete thread_;
    thread_ = nullptr;
  }
}


template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void TreeTraversalThread<GameState, Tensorizor>::add_dirichlet_noise(LocalPolicyArray& P) {
  int rows = P.rows();
  double alpha = manager_params_->dirichlet_alpha_factor / sqrt(rows);
  LocalPolicyArray noise = dirichlet_gen().template generate<LocalPolicyArray>(rng(), alpha, rows);
  P = (1.0 - manager_params_->dirichlet_mult) * P + manager_params_->dirichlet_mult * noise;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void TreeTraversalThread<GameState, Tensorizor>::backprop(const ValueArray& value,
                                                                 BackpropMode mode) {
  profiler_.record(TreeTraversalThreadRegion::kBackprop);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << search_path_str() << " " << value.transpose() << std::endl;
  }

  Node* last_node = search_path_.back().node;
  edge_t* last_edge = search_path_.back().edge;
  if (mode == kTerminal) {
    last_node->update_stats(RealIncrementAndDeduceCertainOutcomes(value), traversal_mode_);
  } else if (mode == kNonterminal) {
    last_node->update_stats(RealIncrement{}, traversal_mode_);
  } else {
    throw util::Exception("Unknown BackpropMode: %d", (int)mode);
  }
  if (last_edge) last_edge->increment_count(traversal_mode_);

  for (int i = search_path_.size() - 2; i >= 0; --i) {
    Node* child = search_path_[i].node;
    edge_t* edge = search_path_[i].edge;
    child->update_stats(RealIncrement{}, traversal_mode_);
    if (i) edge->increment_count(traversal_mode_);
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void TreeTraversalThread<GameState, Tensorizor>::short_circuit_backprop(edge_t* last_edge) {
  // short-circuit
  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << search_path_str() << std::endl;
  }

  last_edge->increment_count(traversal_mode_);

  for (int i = search_path_.size() - 1; i >= 0; --i) {
    Node* child = search_path_[i].node;
    edge_t* edge = search_path_[i].edge;
    child->update_stats(RealIncrement{}, traversal_mode_);
    if (i) edge->increment_count(traversal_mode_);
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
std::string TreeTraversalThread<GameState, Tensorizor>::search_path_str() const {
  std::string delim = GameState::action_delimiter();
  std::vector<std::string> vec;
  for (int n = 1; n < (int)search_path_.size(); ++n) {  // skip the first node
    const GameState& prev_state = search_path_[n - 1].node->stable_data().state;
    Action action = search_path_[n].edge->action();
    vec.push_back(prev_state.action_to_str(action));
  }
  return util::create_string("[%s]", boost::algorithm::join(vec, delim).c_str());
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
core::action_index_t TreeTraversalThread<GameState, Tensorizor>::get_best_action_index(
    Node* tree, NNEvaluation* evaluation) {
  profiler_.record(TreeTraversalThreadRegion::kPUCT);

  PUCTStats stats(*manager_params_, *search_params_, traversal_mode_, tree,
                  tree == shared_data_->root_node.get());

  using PVec = LocalPolicyArray;

  const PVec& P = stats.P;
  const PVec& N = stats.N;
  const PVec& VN = stats.VN;
  PVec& PUCT = stats.PUCT;

  bool add_noise = !search_params_->disable_exploration && manager_params_->dirichlet_mult > 0;
  if (manager_params_->forced_playouts && add_noise) {
    PVec n_forced = (P * manager_params_->k_forced * N.sum()).sqrt();
    auto F1 = (N < n_forced).template cast<dtype>();
    auto F2 = (N > 0).template cast<dtype>();
    auto F = F1 * F2;
    PUCT = PUCT * (1 - F) + F * 1e+6;
  }

  int argmax_index;
  PUCT.maxCoeff(&argmax_index);

  if (nn_eval_service_) {
    nn_eval_service_->record_puct_calc(VN.sum() > 0);
  }

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(thread_id());

    const auto& tree_stats = tree->stats(traversal_mode_);
    core::seat_index_t cp = tree->stable_data().current_player;

    using ArrayT1 = Eigen::Array<dtype, 3, kNumPlayers>;
    ArrayT1 A1;
    A1.setZero();
    A1.row(0) = tree_stats.real_avg;
    A1.row(1) = tree_stats.virtualized_avg;
    A1(2, cp) = 1;

    std::ostringstream ss1;
    ss1 << A1;
    std::string s1 = ss1.str();

    std::vector<std::string> s1_lines;
    boost::split(s1_lines, s1, boost::is_any_of("\n"));

    printer << "real_avg: " << s1_lines[0] << std::endl;
    printer << "virt_avg: " << s1_lines[1] << std::endl;

    std::string cp_line = s1_lines[2];
    std::replace(cp_line.begin(), cp_line.end(), '0', ' ');
    std::replace(cp_line.begin(), cp_line.end(), '1', '*');

    printer << "cp:       " << cp_line << std::endl;

    using ScalarT = typename PVec::Scalar;
    constexpr int kNumRows1 = PolicyShape::count;
    constexpr int kNumRows2 = 9;  // P, V, PW, PL, E, N, VN, PUCT, argmax
    constexpr int kNumRows = kNumRows1 + kNumRows2;
    constexpr int kMaxCols = PVec::MaxRowsAtCompileTime;
    using ArrayT2 = Eigen::Array<ScalarT, kNumRows, Eigen::Dynamic, 0, kNumRows, kMaxCols>;

    ArrayT2 A2(kNumRows, P.rows());
    A2.setZero();

    const ActionMask& valid_actions = tree->stable_data().valid_action_mask;
    const bool* data = valid_actions.data();
    int r = 0;
    int c = 0;
    for (int i = 0; i < kNumGlobalActionsBound; ++i) {
      if (!data[i]) continue;
      Action action = eigen_util::unflatten_index(valid_actions, i);
      for (int j = 0; j < kNumRows1; ++j) {
        A2(j, c) = action[j];
      }
      ++c;
    }
    r += kNumRows1;

    A2.row(r++) = P;
    A2.row(r++) = stats.V;
    A2.row(r++) = stats.PW;
    A2.row(r++) = stats.PL;
    A2.row(r++) = stats.E;
    A2.row(r++) = N;
    A2.row(r++) = stats.VN;
    A2.row(r++) = PUCT;
    A2(r, argmax_index) = 1;

    std::ostringstream ss2;
    ss2 << A2;
    std::string s2 = ss2.str();

    std::vector<std::string> s2_lines;
    boost::split(s2_lines, s2, boost::is_any_of("\n"));

    r = 0;
    printer << "move:  " << s2_lines[r++] << std::endl;
    while (r < kNumRows1) {
      printer << "       " << s2_lines[r++] << std::endl;
    }
    printer << "P:     " << s2_lines[r++] << std::endl;
    printer << "V:     " << s2_lines[r++] << std::endl;
    printer << "PW:    " << s2_lines[r++] << std::endl;
    printer << "PL:    " << s2_lines[r++] << std::endl;
    printer << "E:     " << s2_lines[r++] << std::endl;
    printer << "N:     " << s2_lines[r++] << std::endl;
    printer << "VN:    " << s2_lines[r++] << std::endl;
    printer << "PUCT:  " << s2_lines[r++] << std::endl;

    std::string argmax_line = s2_lines[r];
    std::replace(argmax_line.begin(), argmax_line.end(), '0', ' ');
    std::replace(argmax_line.begin(), argmax_line.end(), '1', '*');

    printer << "argmax:" << argmax_line << std::endl;
    printer << "*************" << std::endl;
  }
  return argmax_index;
}

}  // namespace mcts
