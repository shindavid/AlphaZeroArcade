#include <mcts/SearchLog.hpp>

namespace mcts {

template <core::concepts::Game Game>
inline boost::json::object SearchLog<Game>::log_node_t::to_json() const {
  boost::json::object node_json;
  node_json["index"] = index;
  node_json["N"] = N;

  boost::json::array Q_array;
  for (int i = 0; i < Q.size(); ++i) {
    Q_array.push_back(Q[i]);
  }
  node_json["Q"] = Q_array;

  node_json["state"] = state;

  boost::json::array provably_winning_array;
  for (size_t i = 0; i < provably_winning.size(); ++i) {
    provably_winning_array.push_back(provably_winning[i]);
  }
  node_json["provably_winning"] = provably_winning_array;

  boost::json::array provably_losing_array;
  for (size_t i = 0; i < provably_losing.size(); ++i) {
    provably_losing_array.push_back(provably_losing[i]);
  }
  node_json["provably_losing"] = provably_losing_array;

  return node_json;
}

template <core::concepts::Game Game>
inline boost::json::object SearchLog<Game>::log_edge_t::to_json() const {
  boost::json::object edge_json;
  edge_json["index"] = index;
  edge_json["from"] = from;
  edge_json["to"] = to;
  edge_json["E"] = E;
  edge_json["action"] = action;
  return edge_json;
}

template <core::concepts::Game Game>
inline std::string SearchLog<Game>::json_str() {
  std::stringstream ss;
  boost_util::pretty_print(ss, combine_json());
  return ss.str();
};

template <core::concepts::Game Game>
inline std::string SearchLog<Game>::last_graph_json_str() {
  std::stringstream ss;
  boost_util::pretty_print(ss, graphs.back().graph_repr());
  return ss.str();
};

template <core::concepts::Game Game>
void SearchLog<Game>::build_graph(Graph& graph) {
  using State = Game::State;
  using edge_t = mcts::Node<Game>::edge_t;
  using Node = mcts::Node<Game>;
  auto map = shared_data_->lookup_table.map();

  for (auto [key, node_ix] : *map) {
    const Node* node = shared_data_->lookup_table.get_node(node_ix);
    const State* state = node->stable_data().get_state();
    graph.add_node(node_ix, node->stats().RN, node->stats().Q, Game::IO::compact_state_repr(*state),
                   node->stats().provably_winning, node->stats().provably_losing);
    for (int i = 0; i < node->stable_data().num_valid_actions; ++i) {
      edge_t* edge = node->get_edge(i);

      if (edge->child_index == -1) {
        continue;
      }

      int edge_index = i + node->get_first_edge_index();
      graph.add_edge(edge_index, node_ix, edge->child_index, edge->E, edge->action);
    }
  }
}

template <core::concepts::Game Game>
inline void SearchLog<Game>::update() {
  Graph graph;
  build_graph(graph);
  graph.sort_by_index();
  add_graph(graph);
}

template <core::concepts::Game Game>
inline boost::json::object SearchLog<Game>::combine_json() {
  boost::json::array graphs_array;
  for (const auto& graph : graphs) {
    graphs_array.push_back(graph.graph_repr());
  }

  boost::json::object log_json;
  log_json["graphs"] = graphs_array;

  // Convert the JSON object to a string
  return log_json;
}

template <core::concepts::Game Game>
inline boost::json::object SearchLog<Game>::Graph::graph_repr() const {
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

}  // namespace mcts
