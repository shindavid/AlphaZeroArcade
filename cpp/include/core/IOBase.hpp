#pragma once

namespace core {

template <typename Types, typename State>
struct IOBase {
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action) { return std::to_string(action); }
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr) {
      throw std::runtime_error("print_state not implemented");
    }
    static void print_mcts_results(std::ostream&, const Types::Policy& action_policy,
                                   const Types::SearchResults&) {
      throw std::runtime_error("print_mcts_results not implemented");
    }
    static std::string compact_state_repr(const State& state) {
      throw std::runtime_error("compact_state_repr not implemented");
    }
};

}  // namespace core
