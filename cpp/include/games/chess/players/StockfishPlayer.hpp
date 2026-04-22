#pragma once

#include "games/chess/SyzygyTable.hpp"
#include "games/chess/players/UciPlayer.hpp"

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
    return Params{.num_procs = 8, .movetime = -1, .depth = 10, .nodes = -1, .uci_elo = -1};
  }

  StockfishPlayer(UciPool* pool, const Params& params,
                  const ProcParams& proc_params = kDefaultProcParams)
      : UciPlayer(pool, params, build_proc_params(params, proc_params)) {}

 private:
  static ProcParams build_proc_params(const Params& params, ProcParams proc_params) {
    /* WARNING: UCI_LimitStrength (UCI_Elo) is fundamentally incompatible with Syzygy tablebases.
     *
     * When Syzygy tablebases are active and a hit occurs at the root, Tablebases::root_probe()
     * short-circuits the search. It aggressively prunes the rootMoves list to only include
     * moves with perfect theoretical outcomes and assigns them identical, massive artificial
     * scores (e.g., VALUE_TB_WIN or 20000).
     *
     * This completely breaks the mathematical assumptions of the Skill module. The pick_best()
     * algorithm requires a populated multiPV list with a diverse distribution of standard
     * centipawn scores to calculate the probabilistic "push" (delta) for sub-optimal move
     * selection.
     *
     * If tablebases are loaded:
     * 1. The score difference ('delta') between the best and worst moves becomes 0.
     * 2. The probabilistic 'push' calculation collapses.
     * 3. The rootMoves vector may be pruned to a size smaller than the expected multiPV count.
     *
     * Consequently, the move selection loop fails to evaluate a valid mistake and falls through,
     * returning Move::none() and causing the engine to crash or output "bestmove (none)".
     * SyzygyPath should be disabled when simulating human error via UCI_Elo.
     */
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
