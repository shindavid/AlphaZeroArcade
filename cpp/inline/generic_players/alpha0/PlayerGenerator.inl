#include "generic_players/alpha0/PlayerGenerator.hpp"

#include "search/Constants.hpp"
#include "search/TrainingDataWriter.hpp"

#include <format>

namespace generic::alpha0 {

template <search::concepts::Traits Traits, typename PlayerT, search::Mode Mode>
std::vector<std::string> PlayerGeneratorBase<Traits, PlayerT, Mode>::get_types() const {
  if (Mode == search::kCompetition) {
    // We keep MCTS-C for nostalgic reasons
    return {"alpha0-C", "AlphaZero-Competition", "MCTS-C"};
  } else if (Mode == search::kTraining) {
    // We keep MCTS-T for nostalgic reasons
    return {"alpha0-T", "AlphaZero-Training", "MCTS-T"};
  } else {
    throw util::CleanException("Unknown search::Mode: {}", Mode);
  }
}

template <search::concepts::Traits Traits, typename PlayerT, search::Mode Mode>
std::string PlayerGeneratorBase<Traits, PlayerT, Mode>::get_description() const {
  if (Mode == search::kCompetition) {
    return "Competition AlphaZero player";
  } else if (Mode == search::kTraining) {
    return "Training AlphaZero player";
  } else {
    throw util::CleanException("Unknown search::Mode: {}", Mode);
  }
}

}  // namespace generic::alpha0
