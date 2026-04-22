#pragma once

#include "beta0/GameLogFullRecord.hpp"
#include "beta0/GameLogView.hpp"
#include "beta0/TrainingInfo.hpp"
#include "beta0/concepts/SpecConcept.hpp"
#include "core/BasicTypes.hpp"
#include "search/GameLogCommon.hpp"

#include <cstdint>
#include <vector>

/*
 * This module contains various classes used for the reading and writing of beta0 game log files.
 *
 * The file format is identical to alpha0 except for the DATA section, which is written by
 * beta0::GameLogFullRecord::serialize() and contains:
 *
 * [beta0::GameLogCompactRecord]
 * [PolicyTensorData]                    // policy target — variable-sized
 * [ActionValueTensorData]               // AV target — variable-sized
 * [ActionValueTensorData]               // AU target — variable-sized
 * [PolicyTensorData]                    // child N counts (placeholder, valid=false)
 * [ActionValueTensorData]               // child Q (placeholder, valid=false)
 * [ActionValueTensorData]               // child W (placeholder, valid=false)
 *
 * W_target is retroactively computed in add_terminal() via a lambda-discounted backward pass
 * over stored Q_root values (KataGo LoTV formulation, lambda=5/6):
 *
 *   S[T_max] = Q_root[T_max]  (or zero for last position)
 *   S[t] = (1-lambda) * Q_root[t+1] + lambda * S[t+1]   (backward pass, t = T_max-1 .. 0)
 *   W_target[t] = (Q_root[t] - S[t])^2
 */
namespace beta0 {

template <::beta0::concepts::Spec Spec>
class GameReadLog {
 public:
  using Game = Spec::Game;
  using Symmetries = Spec::Symmetries;
  using GameLogView = beta0::GameLogView<Spec>;
  using TrainingTargets = Spec::TrainingTargets::List;
  using NetworkHeads = Spec::NetworkHeads::List;

  using mem_offset_t = search::GameLogCommon::mem_offset_t;
  using frame_index_t = search::GameLogCommon::frame_index_t;

  using TensorEncodings = Spec::TensorEncodings;
  using PolicyShape = TensorEncodings::PolicyEncoding::Shape;
  using ActionValueShape = TensorEncodings::ActionValueEncoding::Shape;
  using GameLogCompactRecord = beta0::GameLogCompactRecord<Spec>;
  using PolicyTensorData = search::TensorData<PolicyShape>;
  using ActionValueTensorData = search::TensorData<ActionValueShape>;

  using InputFrame = Spec::InputFrame;
  using InputEncoder = TensorEncodings::InputEncoder;
  using InputTensor = InputEncoder::Tensor;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using GameResultTensor = GameResultEncoding::Tensor;

  // indicates offsets relative to the start of the GameData region
  struct DataLayout {
    DataLayout(const search::GameLogMetadata&);

    int final_frame;
    int outcome;
    int sampled_indices_start;
    int mem_offsets_start;
    int records_start;
  };

  GameReadLog(const char* filename, int game_index, const search::GameLogMetadata& metadata,
              const char* buffer);

  static search::ShapeInfo* get_input_shapes();
  static search::ShapeInfo* get_target_shapes();
  static search::ShapeInfo* get_head_shapes();

  void load(int row_index, bool apply_symmetry, const std::vector<int>& target_indices,
            float* output_array) const;

  int num_sampled_frames() const { return metadata_.num_samples; }

 private:
  static constexpr int align(int offset) { return search::GameLogCommon::align(offset); }

  int num_frames() const { return metadata_.num_frames; }

  const InputFrame& get_final_frame() const;
  const GameResultTensor& get_outcome() const;
  frame_index_t get_frame_index(int sample_index) const;
  const GameLogCompactRecord& get_record(mem_offset_t mem_offset) const;
  mem_offset_t get_mem_offset(int frame_index) const;

  const char* filename_;
  const int game_index_;
  const search::GameLogMetadata& metadata_;
  const char* buffer_ = nullptr;
  const DataLayout layout_;
};

template <::beta0::concepts::Spec Spec>
class GameWriteLog : public search::GameWriteLogBase {
 public:
  using mem_offset_t = search::GameLogCommon::mem_offset_t;
  using frame_index_t = search::GameLogCommon::frame_index_t;

  using TensorEncodings = Spec::TensorEncodings;
  using PolicyShape = TensorEncodings::PolicyEncoding::Shape;
  using ActionValueShape = TensorEncodings::ActionValueEncoding::Shape;
  using GameLogCompactRecord = beta0::GameLogCompactRecord<Spec>;
  using PolicyTensorData = search::TensorData<PolicyShape>;
  using ActionValueTensorData = search::TensorData<ActionValueShape>;
  using GameLogFullRecord = beta0::GameLogFullRecord<Spec>;
  using full_record_vec_t = std::vector<GameLogFullRecord*>;

  using Game = Spec::Game;
  using TrainingInfo = beta0::TrainingInfo<Spec>;
  using InputFrame = Spec::InputFrame;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using GameResultTensor = GameResultEncoding::Tensor;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using ValueArray = Game::Types::ValueArray;

  GameWriteLog(core::game_id_t id, int64_t start_timestamp);
  ~GameWriteLog();

  void add(const TrainingInfo&);

  GameLogFullRecord* get_full_record(int index) const { return full_records_[index]; }
  size_t size() const { return full_records_.size(); }
  void add_terminal(const InputFrame& frame, const GameResultTensor& outcome);
  bool was_previous_entry_used_for_policy_training() const;
  const InputFrame& final_frame() const { return final_frame_; }
  const GameResultTensor& outcome() const { return outcome_; }
  bool terminal_added() const { return terminal_added_; }

  // GameWriteLogBase virtuals
  int num_positions() const override;
  bool is_complete() const override;
  bool serialize_position(int move_num, std::vector<char>& data_buf) const override;
  void write_final_sections(std::vector<char>& buf) const override;

 private:
  // lambda for LoTV backward pass (per KataGo)
  static constexpr float kLambda = 5.0f / 6.0f;

  full_record_vec_t full_records_;
  InputFrame final_frame_;
  GameResultTensor outcome_;
  bool terminal_added_ = false;
};

}  // namespace beta0

#include "inline/beta0/GameLog.inl"
