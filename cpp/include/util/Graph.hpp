#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/Node.hpp>

template <core::concepts::Game Game>
class Graph {
 protected:
  using edge_pool_index_t = mcts::Node<Game>::edge_pool_index_t;
  using node_pool_index_t = mcts::Node<Game>::node_pool_index_t;
  using ValueArray = Game::Types::ValueArray;

  struct Node {
    node_pool_index_t index;
    int N;
    ValueArray Q;
    std::string state;
  };

  struct Edge {
    edge_pool_index_t index;
    node_pool_index_t from;
    node_pool_index_t to;
    int E;
    core::action_t action;
  };

 public:
  void sort_by_index() {
    std::sort(nodes.begin(), nodes.end(),
              [](const Node& a, const Node& b) { return a.index < b.index; });
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) { return a.index < b.index; });
  }

  void export_graph_to_json(const std::string& filename) {
    sort_by_index();

    std::ostringstream oss;

    // Nodes
    oss << "{\n";
    oss << "  \"nodes\": [\n";
    for (size_t i = 0; i < nodes.size(); ++i) {
      const Node& node = nodes[i];
      oss << "    {\n";
      oss << "      \"index\": " << node.index << ",\n";
      oss << "      \"N\": " << node.N << ",\n";
      oss << "      \"Q\": [" << node.Q[0] << ", " << node.Q[1] << "],\n";
      oss << "      \"state\": \"" << node.state << "\"\n";
      oss << "    }";
      if (i < nodes.size() - 1) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "  ],\n";

    // Edges
    oss << "  \"edges\": [\n";
    for (size_t i = 0; i < edges.size(); ++i) {
      const Edge& edge = edges[i];
      oss << "    {\n";
      oss << "      \"index\": " << edge.index << ",\n";
      oss << "      \"from\": " << edge.from << ",\n";
      oss << "      \"to\": " << edge.to << ",\n";
      oss << "      \"E\": " << edge.E << ",\n";
      oss << "      \"action\": " << edge.action << "\n";
      oss << "    }";
      if (i < edges.size() - 1) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "  ]\n";
    // End JSON
    oss << "}\n";

    // Write to file
    std::ofstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    file << oss.str();
  }

  void add_node(node_pool_index_t index, int N, const ValueArray& Q, const std::string& state) {
    nodes.emplace_back(index, N, Q, state);
  }

  void add_edge(edge_pool_index_t index, node_pool_index_t from, node_pool_index_t to, int E,
               core::action_t action) {
    edges.emplace_back(index, from, to, E, action);
  }

 private:
  std::vector<Node> nodes;
  std::vector<Edge> edges;
};
