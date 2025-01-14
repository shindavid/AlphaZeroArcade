#pragma once
#include <util/EigenUtil.hpp>
#include <Eigen/Core>

namespace core {

template <typename Types, typename State, typename Constants, typename Rules>
struct IOBase {
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action) { return std::to_string(action); }
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr) {
      throw std::runtime_error("print_state not implemented");
    }
    static void print_mcts_results(std::ostream&, const Types::PolicyTensor& action_policy,
                                   const Types::SearchResults&, int num_rows_to_display);
    static std::string compact_state_repr(const State& state) {
      throw std::runtime_error("compact_state_repr not implemented");
    }
};

}  // namespace core

#include <inline/core/IOBase.inl>