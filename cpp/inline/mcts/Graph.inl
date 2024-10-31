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

  }  // namespace mcts
