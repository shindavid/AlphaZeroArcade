#pragma once

#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/TensorizorConcept.hpp>
#include <common/TrainingDataWriter.hpp>
#include <common/players/MctsPlayer.hpp>

#include <vector>

namespace common {

/*
 * A variant of MctsPlayer that exports training data to a file via TrainingDataWriter.
 */
template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
class DataExportingMctsPlayer : public MctsPlayer<GameState_, Tensorizor_> {
public:
  using GameState = GameState_;
  using Tensorizor = Tensorizor_;
  using GameStateTypes = common::GameStateTypes<GameState>;
  using TensorizorTypes = common::TensorizorTypes<Tensorizor>;
  using dtype = typename GameStateTypes::dtype;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using TrainingDataWriter = common::TrainingDataWriter<GameState, Tensorizor>;
  using TensorGroup = typename TrainingDataWriter::TensorGroup;
  using TrainingDataWriterParams = typename TrainingDataWriter::Params;
  using InputTensor = typename TensorizorTypes::InputTensor;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using SymmetryTransform = AbstractSymmetryTransform<GameState, Tensorizor>;

  using base_t = MctsPlayer<GameState, Tensorizor>;
  using Params = base_t::Params;
  using Mcts = base_t::Mcts;
  using MctsResults = base_t::MctsResults;

  template<typename... BaseArgs>
  DataExportingMctsPlayer(const TrainingDataWriterParams& writer_params, BaseArgs&&...);

  void start_game() override;
  void receive_state_change(
      seat_index_t seat, const GameState& state, action_index_t action) override;
  action_index_t get_action(const GameState&, const ActionMask&) override;
  void end_game(const GameState&, const GameOutcome&) override;

protected:
  static PolicyTensor extract_policy(const MctsResults* results);
  void record_position(const GameState& state, const ActionMask& valid_actions, const PolicyTensor& policy);
  void record_opp_reply(const PolicyTensor& policy);

  /*
   * In order to support the opponent-reply auxiliary policy target (see Section 3.4 of the KataGo paper), we need to
   * temporarily store the partially-written TensorGroup(s), to be completed after the opponent's reply. This demands
   * sharing of data across distinct MctsPlayer instances, which we accomplish by using a static vector. This assumes
   * that the MctsPlayer instances live in the same process; this is enforced with a check in GameServerProxy.
   *
   * We use a vector as opposed to a single TensorGroup because we have data augmentation via symmetry transforms, and
   * all the augmented data need to be updated with the opponent's reply.
   *
   * The last move of the game will not have a corresponding opponent reply. We deal with this by writing all zeros
   * for the opponent reply policy in this case. The python training code masks out these rows when computing the
   * loss for the opponent reply.
   */
  struct transform_group_t {
    transform_group_t(SymmetryTransform* t, TensorGroup* g) : transform(t), group(g) {}

    SymmetryTransform* transform;
    TensorGroup* group;
  };
  using transform_group_vec_t = std::vector<transform_group_t>;
  static transform_group_vec_t transform_groups_;
  static constexpr bool kForceFullSearchIfRecordingAsOppReply = false;

  TrainingDataWriter* writer_;
  TrainingDataWriter::GameData_sptr game_data_;
};

}  // namespace common

#include <common/players/inl/DataExportingMctsPlayer.inl>

