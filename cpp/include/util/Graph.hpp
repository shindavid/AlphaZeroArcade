#pragma once

#include <core/concepts/Game.hpp>

#include <sstream>
#include <type_traits>

namespace util {

template <typename Game, typename = std::void_t<>>
struct has_state_repr : std::false_type {};

template <typename Game>
struct has_state_repr<Game, std::void_t<decltype(Game::IO::state_repr(std::declval<const typename Game::State&>()))>>
    : std::true_type {};

template <core::concepts::Game Game>
class Graph {
 protected:
  using ValueArray = Game::Types::ValueArray;

  struct Node {
    int index;
    int N;
    ValueArray Q;
    std::string state;
  };

  struct Edge {
    int index;
    int from;
    int to;
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

  std::string graph_repr() {
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

  void add_node(int index, int N, const ValueArray& Q, const std::string& state) {
    nodes.emplace_back(index, N, Q, state);
  }

  void add_edge(int index, int from, int to, int E,
               core::action_t action) {
    edges.emplace_back(index, from, to, E, action);
  }

 private:
  std::vector<Node> nodes;
  std::vector<Edge> edges;
};

template <core::concepts::Game Game>
class GraphViz {
 public:
  void add_graph(const Graph<Game> graph) { graphs.push_back(graph); }

  std::string combine_json() {
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

  void write_to_json(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    file << combine_json();
  }

 private:
  std::vector<Graph<Game>> graphs;
};
}  // namespace util