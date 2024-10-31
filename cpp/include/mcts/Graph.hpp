#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/SharedData.hpp>

#include <boost/json.hpp>

#include <sstream>
#include <string>

namespace mcts {

template <core::concepts::Game Game>
class Graph {
 protected:
  using ValueArray = Game::Types::ValueArray;
  using node_index_t = int;
  using edge_index_t = int;

  struct Node {
    node_index_t index;
    int N;
    ValueArray Q;
    std::string state;
  };

  struct Edge {
    edge_index_t index;
    node_index_t from;
    node_index_t to;
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

  boost::json::object graph_repr();

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
class SearchLog {
 public:
  SearchLog(const SharedData<Game>* shared_data) : shared_data_(shared_data) {}

  void add_graph(const Graph<Game> graph) { graphs.push_back(graph); }
  void build_graph(Graph<Game>& graph);
  void update();
  boost::json::object combine_json();
  std::string json_str() {return boost::json::serialize(combine_json());};
  void write_json_to_file(const boost::filesystem::path& filename);

 private:
  const SharedData<Game>* shared_data_;
  std::vector<Graph<Game>> graphs;
};

}  // namespace mcts

#include <inline/mcts/Graph.inl>