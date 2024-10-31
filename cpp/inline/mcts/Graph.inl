#include <mcts/Graph.hpp>

namespace mcts {

template <core::concepts::Game Game>
inline std::string Graph<Game>::graph_repr() {
    sort_by_index();

    std::ostringstream oss;

    // Nodes
    oss << "  {\n";
    oss << "    \"nodes\": [\n";
    for (size_t i = 0; i < nodes.size(); ++i) {
      const Node& node = nodes[i];
      oss << "      {\n";
      oss << "        \"index\": " << node.index << ",\n";
      oss << "        \"N\": " << node.N << ",\n";
      oss << "        \"Q\": [" << node.Q[0] << ", " << node.Q[1] << "],\n";
      oss << "        \"state\": \"" << node.state << "\"\n";
      oss << "      }";
      if (i < nodes.size() - 1) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "    ],\n";

    // Edges
    oss << "    \"edges\": [\n";
    for (size_t i = 0; i < edges.size(); ++i) {
      const Edge& edge = edges[i];
      oss << "      {\n";
      oss << "        \"index\": " << edge.index << ",\n";
      oss << "        \"from\": " << edge.from << ",\n";
      oss << "        \"to\": " << edge.to << ",\n";
      oss << "        \"E\": " << edge.E << ",\n";
      oss << "        \"action\": " << edge.action << "\n";
      oss << "      }";
      if (i < edges.size() - 1) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "    ]\n";
    oss << "  }";

    return oss.str();
  }

  template <core::concepts::Game Game>
  inline std::string GraphViz<Game>::combine_json() {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"graphs\": [\n";
    for (size_t i = 0; i < graphs.size(); ++i) {
      oss << graphs[i].graph_repr();
      if (i < graphs.size() - 1) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
  }

  template <core::concepts::Game Game>
  inline void GraphViz<Game>::write_to_json(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    file << combine_json();
  }

  template <core::concepts::Game Game>
  void GraphViz<Game>::build_graph(Graph<Game>& graph) {
    using State = Game::State;
    using edge_t = typename Node<Game>::edge_t;
    auto map = shared_data_.lookup_table.map();

    for (auto [key, node_ix] : *map) {
      Node<Game>* node = shared_data_.lookup_table.get_node(node_ix);
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
  void GraphViz<Game>::build_graph_viz() {
    Graph<Game> graph;
    build_graph(graph);
    add_graph(graph);
  }

  template <core::concepts::Game Game>
  inline void GraphViz<Game>::update() {
    build_graph_viz();
  }

  }  // namespace mcts
