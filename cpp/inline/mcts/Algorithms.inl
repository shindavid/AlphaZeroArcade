#include "mcts/Algorithms.hpp"

#include "mcts/Constants.hpp"
#include "util/Asserts.hpp"
#include "util/FiniteGroups.hpp"

#include <boost/algorithm/string/join.hpp>
#include <spdlog/spdlog.h>

#include <format>

namespace mcts {

template <typename Traits>
void Algorithms<Traits>::pure_backprop(SearchContext& context, const ValueArray& value) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {} {}", "", context.log_prefix_n(), __func__, search_path_str(context),
             fmt::streamed(value.transpose()));
  }

  RELEASE_ASSERT(!context.search_path.empty());
  Node* last_node = context.search_path.back().node;

  last_node->update_stats([&] {
    auto& stats = last_node->stats();  // thread-safe because executed under mutex
    stats.init_q(value, true);
    stats.RN++;
  });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().RN++;  // thread-safe because executed under mutex
    });
  }
  validate_search_path(context);
}

template <typename Traits>
void Algorithms<Traits>::virtual_backprop(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, search_path_str(context));
  }

  RELEASE_ASSERT(!context.search_path.empty());
  Node* last_node = context.search_path.back().node;

  last_node->update_stats([&] {
    last_node->stats().VN++;  // thread-safe because executed under mutex
  });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().VN++;  // thread-safe because executed under mutex
    });
  }
  validate_search_path(context);
}

template <typename Traits>
void Algorithms<Traits>::undo_virtual_backprop(SearchContext& context) {
  // NOTE: this is not an exact undo of virtual_backprop(), since the context.search_path is
  // modified in between the two calls.

  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, search_path_str(context));
  }

  RELEASE_ASSERT(!context.search_path.empty());

  for (int i = context.search_path.size() - 1; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E--;
      node->stats().VN--;  // thread-safe because executed under mutex
    });
  }
  validate_search_path(context);
}

template <typename Traits>
void Algorithms<Traits>::standard_backprop(SearchContext& context, bool undo_virtual) {
  Node* last_node = context.search_path.back().node;
  auto value = GameResults::to_value_array(last_node->stable_data().VT);

  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {} {}", "", context.log_prefix_n(), __func__, search_path_str(context),
             fmt::streamed(value.transpose()));
  }

  last_node->update_stats([&] {
    auto& stats = last_node->stats();  // thread-safe because executed under mutex
    stats.init_q(value, false);
    stats.RN++;
    stats.VN -= undo_virtual;
  });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E += !undo_virtual;
      auto& stats = node->stats();  // thread-safe because executed under mutex
      stats.RN++;
      stats.VN -= undo_virtual;
    });
  }
  validate_search_path(context);
}

template <typename Traits>
void Algorithms<Traits>::short_circuit_backprop(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (mcts::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, search_path_str(context));
  }

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().RN++;  // thread-safe because executed under mutex
    });
  }
  validate_search_path(context);
}

template <typename Traits>
void Algorithms<Traits>::validate_search_path(const SearchContext& context) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;

  int N = context.search_path.size();
  for (int i = N - 1; i >= 0; --i) {
    context.search_path[i].node->validate_state();
  }
}

template <typename Traits>
std::string Algorithms<Traits>::search_path_str(const SearchContext& context) {
  group::element_t cur_sym = context.root_canonical_sym;
  std::string delim = IO::action_delimiter();
  std::vector<std::string> vec;
  for (const Visitation& visitation : context.search_path) {
    if (!visitation.edge) continue;
    core::action_mode_t mode = visitation.node->action_mode();
    core::action_t action = visitation.edge->action;
    Symmetries::apply(action, cur_sym, mode);
    cur_sym = SymmetryGroup::compose(cur_sym, SymmetryGroup::inverse(visitation.edge->sym));
    vec.push_back(IO::action_to_str(action, mode));
  }
  RELEASE_ASSERT(cur_sym == context.leaf_canonical_sym,
                 "cur_sym={} leaf_canonical_sym={}", cur_sym, context.leaf_canonical_sym);
  return std::format("[{}]", boost::algorithm::join(vec, delim));
}

}  // namespace mcts
