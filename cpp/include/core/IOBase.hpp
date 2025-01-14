#pragma once
#include <util/EigenUtil.hpp>
#include <Eigen/Core>

namespace core {

template <typename Types, typename Rules>
struct IOBase {
    using State = Types::State;
    using Constants = Types::GameConstants;
    using ActionToStrFunc = std::function<std::string(core::action_t, core::action_mode_t)>;
    static std::string action_delimiter() { return "-"; }
    static std::string action_to_str(core::action_t action) { return std::to_string(action); }
    static void print_state(std::ostream&, const State&, core::action_t last_action = -1,
                            const Types::player_name_array_t* player_names = nullptr) {
      throw std::runtime_error("print_state not implemented");
    }
    static void print_mcts_results(std::ostream&, const Types::PolicyTensor&,
                                   const Types::SearchResults&, ActionToStrFunc,
                                   int num_rows_to_display);
    static std::string compact_state_repr(const State& state) {
      throw std::runtime_error("compact_state_repr not implemented");
    }
};

}  // namespace core

#include <inline/core/IOBase.inl>