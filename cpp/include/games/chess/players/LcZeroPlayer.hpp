#pragma once

#include "games/chess/players/UciPlayer.hpp"

namespace a0achess {

class LcZeroPlayer : public UciPlayer {
 public:
  static constexpr std::string_view kTypeStr = "lc0";
  static constexpr std::string_view kDescription = "lc0 player";

  static std::string default_name(const Params& params) {
    return std::format("lc0-{}", params.nodes);
  }

  static inline const ProcParams kDefaultProcParams = {
    .cmd = "extra_deps/lc0/lc0",
    .extra_args = "--weights=extra_deps/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz"
  };

  static Params default_params() {
    return Params{.num_procs = 5, .movetime = -1, .depth = -1, .nodes = 1200};
  }

  LcZeroPlayer(UciPool* pool, const Params& params,
               const ProcParams& proc_params = kDefaultProcParams)
      : UciPlayer(pool, params, proc_params) {}
};

}  // namespace a0achess
