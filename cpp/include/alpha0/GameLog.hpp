#pragma once

#include "alpha0/GameLogFullRecord.hpp"
#include "alpha0/GameLogView.hpp"
#include "alpha0/TrainingInfo.hpp"
#include "alpha0/concepts/SpecConcept.hpp"
#include "core/BasicTypes.hpp"
#include "search/GameLogCommon.hpp"

#include <cstdint>
#include <vector>

/*
 * This module contains various classes used for the reading and writing of alpha0 game log files.
 *
 * Each game log file contains a variable number of games. It is structured as follows:
 *
 * [GameLogFileHeader]    // header
 * [GameLogMetadata...]   // one per game
 * [GameData...]          // one per game
 *
 * The GameData object doesn't correspond to a particular struct; it cannot, since some of the
 * fields are variable-sized. Its layout on disk looks like this:
 *
 *   [InputFrame]         // final frame
 *   [GameResultTensor]   // game result
 *   [pos_index_t...]     // indices of sampled positions
 *   [mem_offset_t...]    // memory-offsets into the DATA region
 *   [DATA]               // differently-sized sections of data
 *
 * The corresponding GameLogMetadata tells us how large each of these sections are.
 *
 * The header tells us how many games are in the file, which allows us to compute the start of the
 * GameData section. Each GameLogMetadata gives us offset information that allows us to seek
 * directly to its corresponding GameData section.
 *
 * The DATA section is written to by Algorithms::serialize_record(), which is specialized per
 * paradigm (currently only alpha0). Its layout looks like:
 *
 * [alpha0::GameLogCompactRecord]
 * [PolicyTensorData]       // policy - variable-sized
 * [ActionValueTensorData]  // action_values - variable-sized
 *
 * See search::TensorData<Shape> for details on the *TensorData encoding.
 */
namespace alpha0 {

template <::alpha0::concepts::Spec Spec>
class GameReadLog {
 public:
  using Game = Spec::Game;
  using Symmetries = Spec::Symmetries;
  using GameLogView = alpha0::GameLogView<Spec>;
  using TrainingTargets = Spec::TrainingTargets::List;
  using NetworkHeads = Spec::NetworkHeads::List;

  using mem_offset_t = search::GameLogCommon::mem_offset_t;
  using frame_index_t = search::GameLogCommon::frame_index_t;

  using TensorEncodings = Spec::TensorEncodings;
  using PolicyShape = TensorEncodings::PolicyEncoding::Shape;
  using ActionValueShape = TensorEncodings::ActionValueEncoding::Shape;
  using GameLogCompactRecord = alpha0::GameLogCompactRecord<Spec>;
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

template <::alpha0::concepts::Spec Spec>
class GameWriteLog : public search::GameWriteLogBase {
 public:
  using mem_offset_t = search::GameLogCommon::mem_offset_t;
  using frame_index_t = search::GameLogCommon::frame_index_t;

  using TensorEncodings = Spec::TensorEncodings;
  using PolicyShape = TensorEncodings::PolicyEncoding::Shape;
  using ActionValueShape = TensorEncodings::ActionValueEncoding::Shape;
  using GameLogCompactRecord = alpha0::GameLogCompactRecord<Spec>;
  using PolicyTensorData = search::TensorData<PolicyShape>;
  using ActionValueTensorData = search::TensorData<ActionValueShape>;
  using GameLogFullRecord = alpha0::GameLogFullRecord<Spec>;
  using full_record_vec_t = std::vector<GameLogFullRecord*>;

  using Game = Spec::Game;
  using TrainingInfo = alpha0::TrainingInfo<Spec>;
  using InputFrame = Spec::InputFrame;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using GameResultTensor = GameResultEncoding::Tensor;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;

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
  full_record_vec_t full_records_;
  InputFrame final_frame_;
  GameResultTensor outcome_;
  bool terminal_added_ = false;
};

}  // namespace alpha0

#include "inline/alpha0/GameLog.inl"
