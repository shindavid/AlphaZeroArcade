#pragma once

#include "core/BasicTypes.hpp"
#include "search/GameLogBase.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

#include <cstdint>
#include <vector>

/*
 * This module contains various classes used for the reading and writing of game log files.
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
 * paradigm (e.g., alpha0, beta0). For alpha0, its layout looks like:
 *
 * [alpha0::GameLogCompactRecord]
 * [PolicyTensorData]       // policy - variable-sized
 * [ActionValueTensorData]  // action_values - variable-sized
 *
 * See search::TensorData<Shape> for details on the *TensorData encoding.
 */
namespace search {

template <search::concepts::SearchSpec SearchSpec>
class GameReadLog : public GameLogBase<SearchSpec> {
 public:
  using Game = SearchSpec::Game;
  using EvalSpec = SearchSpec::EvalSpec;
  using Symmetries = EvalSpec::Symmetries;
  using GameLogView = SearchSpec::GameLogView;
  using TrainingTargets = EvalSpec::TrainingTargets::List;
  using NetworkHeads = EvalSpec::NetworkHeads::List;

  using mem_offset_t = GameLogCommon::mem_offset_t;
  using frame_index_t = GameLogCommon::frame_index_t;

  using GameLogBase = search::GameLogBase<SearchSpec>;
  using GameLogCompactRecord = GameLogBase::GameLogCompactRecord;
  using PolicyTensorData = GameLogBase::PolicyTensorData;
  using ActionValueTensorData = GameLogBase::ActionValueTensorData;

  using InputFrame = EvalSpec::InputFrame;
  using TensorEncodings = EvalSpec::TensorEncodings;
  using InputEncoder = TensorEncodings::InputEncoder;
  using InputTensor = InputEncoder::Tensor;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using GameResultTensor = GameResultEncoding::Tensor;

  // indicates offsets relative to the start of the GameData region
  struct DataLayout {
    DataLayout(const GameLogMetadata&);

    int final_frame;
    int outcome;
    int sampled_indices_start;
    int mem_offsets_start;
    int records_start;
  };

  GameReadLog(const char* filename, int game_index, const GameLogMetadata& metadata,
              const char* buffer);

  static ShapeInfo* get_input_shapes();
  static ShapeInfo* get_target_shapes();
  static ShapeInfo* get_head_shapes();

  void load(int row_index, bool apply_symmetry, const std::vector<int>& target_indices,
            float* output_array) const;

  int num_sampled_frames() const { return metadata_.num_samples; }

 private:
  static constexpr int align(int offset) { return GameLogCommon::align(offset); }

  int num_frames() const { return metadata_.num_frames; }

  const InputFrame& get_final_frame() const;
  const GameResultTensor& get_outcome() const;
  frame_index_t get_frame_index(int sample_index) const;
  const GameLogCompactRecord& get_record(mem_offset_t mem_offset) const;
  mem_offset_t get_mem_offset(int frame_index) const;

  const char* filename_;
  const int game_index_;
  const GameLogMetadata& metadata_;
  const char* buffer_ = nullptr;
  const DataLayout layout_;
};

template <search::concepts::SearchSpec SearchSpec>
class GameLogSerializer;  // Forward declaration

template <search::concepts::SearchSpec SearchSpec>
class GameWriteLog : public GameLogBase<SearchSpec> {
 public:
  friend class GameLogSerializer<SearchSpec>;
  using mem_offset_t = GameLogCommon::mem_offset_t;
  using frame_index_t = GameLogCommon::frame_index_t;

  using GameLogBase = search::GameLogBase<SearchSpec>;
  using GameLogCompactRecord = GameLogBase::GameLogCompactRecord;
  using PolicyTensorData = GameLogBase::PolicyTensorData;
  using ActionValueTensorData = GameLogBase::ActionValueTensorData;
  using GameLogFullRecord = GameLogBase::GameLogFullRecord;
  using full_record_vec_t = GameLogBase::full_record_vec_t;

  using Game = SearchSpec::Game;
  using EvalSpec = SearchSpec::EvalSpec;
  using TrainingInfo = SearchSpec::TrainingInfo;
  using InputFrame = SearchSpec::EvalSpec::InputFrame;
  using TensorEncodings = EvalSpec::TensorEncodings;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using GameResultTensor = GameResultEncoding::Tensor;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;

  GameWriteLog(core::game_id_t id, int64_t start_timestamp);
  ~GameWriteLog();

  void add(const TrainingInfo&);

  void add_terminal(const InputFrame& frame, const GameResultTensor& outcome);
  bool was_previous_entry_used_for_policy_training() const;
  int sample_count() const { return sample_count_; }
  core::game_id_t id() const { return id_; }
  int64_t start_timestamp() const { return start_timestamp_; }

 private:
  full_record_vec_t full_records_;
  InputFrame final_frame_;
  GameResultTensor outcome_;
  const core::game_id_t id_;
  const int64_t start_timestamp_;
  int sample_count_ = 0;
  bool terminal_added_ = false;
};

/*
 * Class used to serialize GameWriteLog objects into a char buffer.
 *
 * The reason we have this class, rather than making serialize() a member function of GameWriteLog,
 * is so that the various std::vector variables used in serialization can be allocated once and
 * reused across multiple GameWriteLog objects.
 */
template <search::concepts::SearchSpec SearchSpec>
class GameLogSerializer {
 public:
  using Game = SearchSpec::Game;
  using frame_index_t = GameLogCommon::frame_index_t;
  using mem_offset_t = GameLogCommon::mem_offset_t;
  using GameLogBase = search::GameLogBase<SearchSpec>;
  using GameWriteLog = search::GameWriteLog<SearchSpec>;

  using GameLogCompactRecord = GameLogBase::GameLogCompactRecord;
  using PolicyTensorData = GameLogBase::PolicyTensorData;
  using ActionValueTensorData = GameLogBase::ActionValueTensorData;
  using GameLogFullRecord = GameLogBase::GameLogFullRecord;

  GameLogMetadata serialize(const GameWriteLog* log, std::vector<char>& buf, int client_id);

 private:
  std::vector<frame_index_t> sampled_indices_;
  std::vector<mem_offset_t> mem_offsets_;
  std::vector<char> data_buf_;
};

}  // namespace search

#include "inline/search/GameLog.inl"
