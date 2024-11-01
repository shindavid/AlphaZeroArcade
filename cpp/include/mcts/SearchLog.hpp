#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/SharedData.hpp>

#include <boost/json.hpp>

#include <sstream>
#include <string>

namespace mcts {

template <core::concepts::Game Game>
class SearchLog {
 protected:
  using ValueArray = Game::Types::ValueArray;
  using node_index_t = int;
  using edge_index_t = int;

 public:
  SearchLog(const SharedData<Game>* shared_data) : shared_data_(shared_data) {}

  struct log_node_t {
    node_index_t index;
    int N;
    ValueArray Q;
    std::string state;
  };

  struct log_edge_t {
    edge_index_t index;
    node_index_t from;
    node_index_t to;
    int E;
    core::action_t action;
  };

  class Graph {
   public:
    void sort_by_index() {
      std::sort(nodes.begin(), nodes.end(),
                [](const log_node_t& a, const log_node_t& b) { return a.index < b.index; });
      std::sort(edges.begin(), edges.end(),
                [](const log_edge_t& a, const log_edge_t& b) { return a.index < b.index; });
    }

    boost::json::object graph_repr() const;

    void add_node(node_index_t index, int N, const ValueArray& Q, const std::string& state) {
      nodes.emplace_back(index, N, Q, state);
    }

    void add_edge(edge_index_t index, node_index_t from, node_index_t to, int E,
                  core::action_t action) {
      edges.emplace_back(index, from, to, E, action);
    }

   private:
    std::vector<log_node_t> nodes;
    std::vector<log_edge_t> edges;
  };

  void update();
  std::string json_str() {return boost::json::serialize(combine_json());};
  void write_json_to_file(const boost::filesystem::path& filename);

 private:
  const SharedData<Game>* shared_data_;
  std::vector<Graph> graphs;

  void add_graph(const Graph& graph) { graphs.push_back(graph); }
  void build_graph(Graph& graph);
  boost::json::object combine_json();
};

}  // namespace mcts

#include <inline/mcts/SearchLog.inl>