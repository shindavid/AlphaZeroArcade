#include <connect4/C4NNetPlayer.hpp>

#include <util/RepoUtil.hpp>

namespace c4 {

inline NNetPlayer::Params::Params()
  : model_filename(util::Repo::root() / "c4_model.pt") {}

inline NNetPlayer::NNetPlayer(const Params& params)
  : base_t("CPU")
  , params_(params)
  , net_(params.model_filename)
{
  throw std::exception();
}

inline void NNetPlayer::start_game(const player_array_t& players, common::player_index_t seat_assignment) {
  throw std::exception();
}

inline void NNetPlayer::receive_state_change(
    common::player_index_t, const GameState&, common::action_index_t, const Result&)
{
  throw std::exception();
}

inline common::action_index_t NNetPlayer::get_action(const GameState&, const ActionMask&) {
  throw std::exception();
}

}  // namespace c4
