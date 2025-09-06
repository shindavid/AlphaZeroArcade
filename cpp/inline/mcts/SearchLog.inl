#include "a0/SearchLog.hpp"

#include "util/BitSet.hpp"
#include "util/BoostUtil.hpp"

namespace a0 {

template <search::concepts::GraphTraits GraphTraits>
inline boost::json::object SearchLog<GraphTraits>::LogNode::to_json() const {
  boost::json::object node_json;
  node_json["index"] = index;
  node_json["N"] = N;

  boost::json::array Q_array;
  for (int i = 0; i < Q.size(); ++i) {
    Q_array.push_back(Q[i]);
  }
  node_json["Q"] = Q_array;

  node_json["state"] = state;
  node_json["provably_winning"] = bitset_util::to_string(provably_winning);
  node_json["provably_losing"] = bitset_util::to_string(provably_losing);
  node_json["active_seat"] = active_seat;
  return node_json;
}

template <search::concepts::GraphTraits GraphTraits>
inline boost::json::object SearchLog<GraphTraits>::LogEdge::to_json() const {
  boost::json::object edge_json;
  edge_json["index"] = index;
  edge_json["from"] = from;
  edge_json["to"] = to;
  edge_json["E"] = E;
  edge_json["action"] = action;
  return edge_json;
}

template <search::concepts::GraphTraits GraphTraits>
inline std::string SearchLog<GraphTraits>::json_str() {
  std::stringstream ss;
  boost_util::pretty_print(ss, combine_json());
  return ss.str();
};

template <search::concepts::GraphTraits GraphTraits>
inline std::string SearchLog<GraphTraits>::last_graph_json_str() {
  std::stringstream ss;
  boost_util::pretty_print(ss, graphs_.back().graph_repr());
  return ss.str();
};

template <search::concepts::GraphTraits GraphTraits>
void SearchLog<GraphTraits>::build_graph(Graph& graph) {
  auto map = lookup_table_->map();

  for (auto [key, node_ix] : *map) {
    const Node* node = lookup_table_->get_node(node_ix);
    const State* state = node->stable_data().get_state();
    const auto stats = node->stats_safe();  // make a copy
    graph.add_node(node_ix, stats.RN, stats.Q, Game::IO::compact_state_repr(*state),
                   stats.provably_winning, stats.provably_losing, node->stable_data().active_seat);
    for (int i = 0; i < node->stable_data().num_valid_actions; ++i) {
      Edge* edge = lookup_table_->get_edge(node, i);

      if (edge->child_index == -1) {
        continue;
      }

      int edge_index = i + node->get_first_edge_index();
      graph.add_edge(edge_index, node_ix, edge->child_index, edge->E, edge->action);
    }
  }
}

template <search::concepts::GraphTraits GraphTraits>
inline void SearchLog<GraphTraits>::update() {
  Graph graph;
  build_graph(graph);
  graph.sort_by_index();
  add_graph(graph);
}

template <search::concepts::GraphTraits GraphTraits>
inline boost::json::object SearchLog<GraphTraits>::combine_json() {
  boost::json::array graphs_array;
  for (const auto& graph : graphs_) {
    graphs_array.push_back(graph.graph_repr());
  }

  boost::json::object log_json;
  log_json["graphs"] = graphs_array;

  // Convert the JSON object to a string
  return log_json;
}

template <search::concepts::GraphTraits GraphTraits>
inline boost::json::object SearchLog<GraphTraits>::Graph::graph_repr() const {
  boost::json::object graph_json;

  // Nodes
  boost::json::array nodes_array;
  for (const auto& node : nodes) {
    nodes_array.push_back(node.to_json());
  }
  graph_json["nodes"] = nodes_array;

  // Edges
  boost::json::array edges_array;
  for (const auto& edge : edges) {
    edges_array.push_back(edge.to_json());
  }
  graph_json["edges"] = edges_array;

  return graph_json;
}

}  // namespace a0
