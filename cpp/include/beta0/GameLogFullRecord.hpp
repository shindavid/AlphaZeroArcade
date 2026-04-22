#pragma once

#include "beta0/BackupSampleData.hpp"
#include "beta0/GameLogCompactRecord.hpp"
#include "beta0/concepts/SpecConcept.hpp"
#include "core/BasicTypes.hpp"
#include "search/GameLogCommon.hpp"

#include <vector>

namespace beta0 {

template <beta0::concepts::Spec Spec>
struct TrainingInfo;

template <beta0::concepts::Spec Spec>
struct GameLogFullRecord {
  using InputFrame = Spec::InputFrame;
  using TensorEncodings = Spec::TensorEncodings;
  using InputEncoder = TensorEncodings::InputEncoder;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using ValueArray = Spec::Game::Types::ValueArray;
  using Move = Spec::Game::Move;

  using PolicyShape = TensorEncodings::PolicyEncoding::Shape;
  using ActionValueShape = TensorEncodings::ActionValueEncoding::Shape;
  using PolicyTensorData = search::TensorData<PolicyShape>;
  using ActionValueTensorData = search::TensorData<ActionValueShape>;
  using GameLogCompactRecord = beta0::GameLogCompactRecord<Spec>;

  GameLogFullRecord() = default;
  explicit GameLogFullRecord(const TrainingInfo<Spec>&);
  void serialize(std::vector<char>& buf) const;

  // Called by GameWriteLog::add_terminal() to retroactively set W_target.
  void set_W_target(const ValueArray& W_target_val);

  InputFrame frame;
  PolicyTensor policy_target;                   // only valid if policy_target_valid
  ActionValueTensor action_values;              // AV target, only valid if action_values_valid
  ActionValueTensor action_values_uncertainty;  // AU target, only valid if action_values_valid
  ValueArray Q_root;                            // search Q at this step; used for W computation
  ValueArray W_target;                          // retroactively filled
  Move move;
  core::seat_index_t active_seat;
  bool use_for_training;
  bool policy_target_valid;
  bool action_values_valid;
  bool W_target_valid = false;

  BackupSampleData<Spec> backup_sample;
};

}  // namespace beta0

#include "inline/beta0/GameLogFullRecord.inl"
