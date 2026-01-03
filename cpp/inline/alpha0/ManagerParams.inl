#include "alpha0/ManagerParams.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
inline ManagerParams<EvalSpec>::ManagerParams(search::Mode m) : Base(m) {
  if (m == search::kCompetition) {
    dirichlet_mult = 0;
    dirichlet_alpha_factor = 0;
    forced_playouts = false;
    starting_root_softmax_temperature = 1;
    ending_root_softmax_temperature = 1;
    root_softmax_temperature_half_life = 1;
  } else if (m == search::kTraining) {
  } else {
    throw util::Exception("Unknown search::Mode: {}", m);
  }
}

template <core::concepts::EvalSpec EvalSpec>
inline auto ManagerParams<EvalSpec>::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("Manager options");

  auto out =
    desc.template add_option<"cpuct", 'c'>(po2::default_value("{:.2f}", &cPUCT), "cPUCT value")
      .template add_option<"dirichlet-mult", 'd'>(po2::default_value("{:.2f}", &dirichlet_mult),
                                                  "dirichlet mult")
      .template add_hidden_option<"starting-root-softmax-temp">(
        po::value<float>(&starting_root_softmax_temperature)
          ->default_value(starting_root_softmax_temperature),
        "starting root softmax temperature")
      .template add_hidden_option<"ending-root-softmax-temp">(
        po::value<float>(&ending_root_softmax_temperature)
          ->default_value(ending_root_softmax_temperature),
        "ending root softmax temperature")
      .template add_hidden_option<"root-softmax-temp-half-life">(
        po::value<float>(&root_softmax_temperature_half_life)
          ->default_value(root_softmax_temperature_half_life),
        "root softmax temperature half-life")
      .template add_option<"dirichlet-alpha-factor">(
        po2::default_value("{:.2f}", &dirichlet_alpha_factor), "dirichlet alpha factor")
      .template add_hidden_flag<"forced-playouts", "no-forced-playouts">(
        &forced_playouts, "enable forced playouts", "disable forced playouts")
      .template add_hidden_flag<"enable-first-play-urgency", "disable-first-play-urgency">(
        &enable_first_play_urgency, "enable first play urgency", "disable first play urgency");

  return out.add(Base::make_options_description());
}

}  // namespace alpha0
