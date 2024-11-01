#include <mcts/SearchLog.hpp>

namespace mcts {

template <core::concepts::Game Game>
void SearchLog<Game>::build_graph(Graph<Game>& graph) {
  using State = Game::State;
  using edge_t = typename Node<Game>::edge_t;
  auto map = shared_data_->lookup_table.map();

  for (auto [key, node_ix] : *map) {
    const Node<Game>* node = shared_data_->lookup_table.get_node(node_ix);
    const State* state = node->stable_data().get_state();
    graph.add_node(node_ix, node->stats().RN, node->stats().Q,
                   Game::IO::compact_state_repr(*state));
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
  Graph<Game> graph;
  build_graph(graph);
  add_graph(graph);
}

template <core::concepts::Game Game>
inline boost::json::object SearchLog<Game>::combine_json() {
  boost::json::array graphs_array;
  for (auto graph : graphs) {
    graphs_array.push_back(graph.graph_repr());
  }

  boost::json::object log_json;
  log_json["graphs"] = graphs_array;

  // Convert the JSON object to a string
  return log_json;
}

template <core::concepts::Game Game>
inline void SearchLog<Game>::write_json_to_file(const boost::filesystem::path& filename) {
  std::ofstream file(filename);  // Convert path to string for ofstream
  if (file.is_open()) {
    file << json_str();
    file.close();
  } else {
    throw std::runtime_error("Unable to open file: " + filename.string());
  }
}

template <core::concepts::Game Game>
inline boost::json::object Graph<Game>::graph_repr() {
  sort_by_index();

  boost::json::object graph_json;

  // Nodes
  boost::json::array nodes_array;
  for (Node& node : nodes) {
    boost::json::object node_obj;
    node_obj["index"] = node.index;
    node_obj["N"] = node.N;
    node_obj["Q"] = boost::json::array{node.Q[0], node.Q[1]};
    node_obj["state"] = node.state;

    nodes_array.push_back(node_obj);
  }
  graph_json["nodes"] = nodes_array;

  // Edges
  boost::json::array edges_array;
  for (Edge& edge : edges) {
    boost::json::object edge_obj;
    edge_obj["index"] = edge.index;
    edge_obj["from"] = edge.from;
    edge_obj["to"] = edge.to;
    edge_obj["E"] = edge.E;
    edge_obj["action"] = edge.action;

    edges_array.push_back(edge_obj);
  }
  graph_json["edges"] = edges_array;

  return graph_json;
}

}  // namespace mcts
