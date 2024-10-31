#pragma once

#include <core/concepts/Game.hpp>

#include <sstream>
#include <type_traits>

namespace mcts {

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

  std::string graph_repr();

  void add_node(int index, int N, const ValueArray& Q, const std::string& state) {
    nodes.emplace_back(index, N, Q, state);
  }

  void add_edge(int index, int from, int to, int E, core::action_t action) {
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

  std::string combine_json();

  void write_to_json(const std::string& filename);

 private:
  std::vector<Graph<Game>> graphs;
};
}  // namespace util

#include <inline/mcts/Graph.inl>