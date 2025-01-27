#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/SharedData.hpp>
#include <util/BoostUtil.hpp>

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
  using player_bitset_t = Game::Types::player_bitset_t;

 public:
  SearchLog(const SharedData<Game>* shared_data) : shared_data_(shared_data) {}

  struct LogNode {
    node_index_t index;
    int N;
    ValueArray Q;
    std::string state;
    player_bitset_t provably_winning;
    player_bitset_t provably_losing;
    core::seat_index_t active_seat;

    boost::json::object to_json() const;
  };

  struct LogEdge {
    edge_index_t index;
    node_index_t from;
    node_index_t to;
    int E;
    core::action_t action;
    boost::json::object to_json() const;
  };

  class Graph {
   public:
    void sort_by_index() {
      std::sort(nodes.begin(), nodes.end(),
                [](const LogNode& a, const LogNode& b) { return a.index < b.index; });
      std::sort(edges.begin(), edges.end(),
                [](const LogEdge& a, const LogEdge& b) { return a.index < b.index; });
    }

    boost::json::object graph_repr() const;

    void add_node(node_index_t index, int N, const ValueArray& Q, const std::string& state,
                  const player_bitset_t& provably_winning, const player_bitset_t& provably_losing,
                  core::seat_index_t active_seat) {
      nodes.emplace_back(index, N, Q, state, provably_winning, provably_losing, active_seat);
    }

    void add_edge(edge_index_t index, node_index_t from, node_index_t to, int E,
                  core::action_t action) {
      edges.emplace_back(index, from, to, E, action);
    }

   private:
    std::vector<LogNode> nodes;
    std::vector<LogEdge> edges;
  };

  void update();
  std::string json_str();
  std::string last_graph_json_str();

 private:
  const SharedData<Game>* shared_data_;
  std::vector<Graph> graphs;

  void add_graph(const Graph& graph) { graphs.push_back(graph); }
  void build_graph(Graph& graph);
  boost::json::object combine_json();
};

}  // namespace mcts

#include <inline/mcts/SearchLog.inl>