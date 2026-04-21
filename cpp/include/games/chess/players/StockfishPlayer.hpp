#pragma once

#include "games/chess/players/UciPlayer.hpp"
#include "games/chess/SyzygyTable.hpp"

#include <format>

namespace a0achess {

class StockfishPlayer : public UciPlayer {
 public:
  static constexpr std::string_view kTypeStr = "stockfish";
  static constexpr std::string_view kDescription = "stockfish player";

  static std::string default_name(const Params& params) {
    return std::format("Stockfish-{}", params.depth);
  }

  inline static const ProcParams kDefaultProcParams = {
    .cmd = "/workspace/repo/extra_deps/stockfish/stockfish-ubuntu-x86-64-avx2",
    .extra_args = "",
    .uci_settings = ""};

  static constexpr Params default_params() {
    return Params{.num_procs = 8, .movetime = -1, .depth = 10, .nodes = -1};
  }

  StockfishPlayer(UciPool* pool, const Params& params,
                  const ProcParams& proc_params = kDefaultProcParams)
      : UciPlayer(pool, params, build_proc_params(params, proc_params)) {}

 private:
  static ProcParams build_proc_params(const Params& params, ProcParams proc_params) {
    if (params.uci_elo > 0) {
      proc_params.uci_settings += "setoption name UCI_LimitStrength value true\n";
      proc_params.uci_settings += std::format("setoption name UCI_Elo value {}\n", params.uci_elo);
    } else {
      proc_params.uci_settings +=
        std::format("setoption name SyzygyPath value {}\n", SyzygyTable::kSyzygyPath);
    }
    return proc_params;
  }
};

}  // namespace a0achess
