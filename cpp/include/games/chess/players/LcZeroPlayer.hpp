#pragma once

#include "games/chess/players/UciPlayer.hpp"
#include "games/chess/SyzygyTable.hpp"

namespace a0achess {

class LcZeroPlayer : public UciPlayer {
 public:
  static constexpr std::string_view kTypeStr = "lc0";
  static constexpr std::string_view kDescription = "lc0 player";

  static std::string default_name(const Params& params) {
    return std::format("lc0-{}", params.nodes);
  }

  inline static const ProcParams kDefaultProcParams = {
    .cmd = "/workspace/repo/extra_deps/lc0/lc0",
    .extra_args =
      "--weights=/workspace/repo/extra_deps/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz",
    .uci_settings = std::format("setoption name SyzygyPath value {}\n", SyzygyTable::kSyzygyPath)};

  static constexpr Params default_params() {
    return Params{.num_procs = 5, .movetime = -1, .depth = -1, .nodes = 1200};
  }

  LcZeroPlayer(UciPool* pool, const Params& params,
               const ProcParams& proc_params = kDefaultProcParams)
      : UciPlayer(pool, params, proc_params) {
    RELEASE_ASSERT(params.uci_elo < 0, "uci_elo option not supported for lc0 player");
  }
};

}  // namespace a0achess
