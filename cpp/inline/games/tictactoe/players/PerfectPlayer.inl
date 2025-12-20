#include "games/tictactoe/players/PerfectPlayer.hpp"

#include "games/tictactoe/PerfectStrategyLookupTable.hpp"
#include "util/Random.hpp"

#include <bit>
#include <bitset>

namespace tictactoe {

inline auto PerfectPlayer::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("tictactoe::PerfectPlayer options");
  return desc
    .template add_option<"strength", 's'>(po::value<int>(&strength)->default_value(strength),
                                          "strength (0-1). 0 is random, 1 is perfect.")
    .template add_option<"verbose", 'v'>(po::bool_switch(&verbose)->default_value(verbose),
                                         "verbose mode");
}

inline PerfectPlayer::PerfectPlayer(const Params& params)
    : params_(params), lookup_map_(make_lookup_map()) {
  CLEAN_ASSERT(params_.strength >= 0 && params_.strength <= 1, "strength must be in [0, 1]");
}

inline PerfectPlayer::ActionResponse PerfectPlayer::get_action_response(
  const ActionRequest& request) {
  if (request.aux) {
    return ActionResponse::make_move(request.aux - 1);
  }

  ActionResponse response = ActionResponse::make_move(get_action_response_helper(request));
  response.set_aux(response.action + 1);
  return response;
}

inline core::action_t PerfectPlayer::get_action_response_helper(
  const ActionRequest& request) {
  const State& state = request.state;
  const ActionMask& valid_actions = request.valid_actions;

  if (params_.strength == 0) {
    return valid_actions.choose_random_on_index();
  }

  // if only one legal move, make it
  if (valid_actions.count() == 1) {
    return valid_actions.choose_random_on_index();
  }

  core::seat_index_t cp = Game::Rules::get_current_player(state);
  mask_t my_mask = state.cur_player_mask;
  mask_t opp_mask = state.opponent_mask();
  mask_t full_mask = my_mask | opp_mask;
  mask_t x_mask = cp == kX ? my_mask : opp_mask;
  mask_t o_mask = cp == kO ? my_mask : opp_mask;

  if (params_.verbose) {
    std::cout << "PerfectPlayer::" << __func__ << std::endl;
    std::cout << "  cp:        " << int(cp) << std::endl;
    std::cout << "  valid:     " << valid_actions.to_string_natural() << std::endl;
    std::cout << "  my_mask:   " << std::bitset<16>(my_mask) << std::endl;
    std::cout << "  opp_mask:  " << std::bitset<16>(opp_mask) << std::endl;
    std::cout << "  full_mask: " << std::bitset<16>(full_mask) << std::endl;
    std::cout << "  x_mask:    " << std::bitset<16>(x_mask) << std::endl;
    std::cout << "  o_mask:    " << std::bitset<16>(o_mask) << std::endl;
  }

  // check for winning move
  for (mask_t mask : Game::kThreeInARowMasks) {
    if ((std::popcount(uint32_t(mask & my_mask))) == 2 && ((mask & opp_mask) == 0)) {
      int a = std::countr_zero(uint32_t(mask & ~full_mask));
      if (params_.verbose) {
        std::cout << "    winning along:  " << std::bitset<16>(mask) << std::endl;
        std::cout << "    winning move:   " << a << std::endl;
      }
      return a;
    }
  }

  // block opponent's winning move
  for (mask_t mask : Game::kThreeInARowMasks) {
    if ((std::popcount(uint32_t(mask & opp_mask))) == 2 && ((mask & my_mask) == 0)) {
      int a = std::countr_zero(uint32_t(mask & ~full_mask));
      if (params_.verbose) {
        std::cout << "    blocking along: " << std::bitset<16>(mask) << std::endl;
        std::cout << "    blocking move:  " << a << std::endl;
      }
      return a;
    }
  }

  // lookup
  uint64_t key = make_lookup(x_mask, o_mask);
  if (params_.verbose) {
    printf("    lookup: %08ux|%08ux\n", uint32_t(key >> 32), uint32_t(key));
  }

  try {
    return lookup_map_.at(key).select();
  } catch (const std::out_of_range&) {
    throw util::Exception("lookup failed ({:08x}|{:08x})", x_mask, o_mask);
  }
}

/*
 * The uint64_t contains the move probability, multiplied by 255 and rounded down to the nearest
 * int, for moves 1-8, one byte per move, with move 1 in the high byte and move 8 in the low byte.
 * The move probability for move 0 is 1 minus the sum of the other move probabilities.
 */
inline PerfectPlayer::policy_t::policy_t(uint64_t u) {
  for (int i = kNumCells - 1; i >= 1; --i) {
    p[i] = (u & 0xff);
    u >>= 8;
  }
  p[0] = 255 - std::accumulate(p + 1, p + kNumCells, 0.0f);

  // divide every element of p by 255:
  for (int i = 0; i < kNumCells; ++i) {
    p[i] /= 255.0f;
  }
}

inline int PerfectPlayer::policy_t::select() const {
  return util::Random::weighted_sample(p, p + kNumCells);
}

inline PerfectPlayer::lookup_map_t PerfectPlayer::make_lookup_map() {
  lookup_map_t lookup_map;

  for (const auto& [key, value] : lookup_table) {
    lookup_map[key] = policy_t(value);
  }

  return lookup_map;
}

inline uint64_t PerfectPlayer::make_lookup(mask_t x_mask, mask_t o_mask) {
  return (uint64_t(x_mask) << 32) | uint64_t(o_mask);
}

}  // namespace tictactoe
