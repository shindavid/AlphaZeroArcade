#pragma once

#include <core/BasicTypes.hpp>
#include <core/Constants.hpp>
#include <core/concepts/Game.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace core {

struct ShapeInfo {
  template <eigen_util::concepts::FTensor Tensor> void init(const char* nm, int target_idx);
  ~ShapeInfo();

  const char* name = nullptr;
  int* dims = nullptr;
  int num_dims = 0;
  int target_index = -1;
};

struct GameLogBase {
  static constexpr int kAlignment = 16;
  static constexpr bool kSingleActionTypeOptimization = true;

  using pos_index_t = int32_t;

  struct tensor_index_t {
    // If start < end, then the representation is sparse, and entries [start, end) of
    // the sparse_tensor_entry_t region represent the sparse tensor.
    // Else, if start == end >= 0, then start is the index into the dense tensor region.
    // Else, if start == end < 0, then there is no policy target.
    // Any other start/end pair is invalid.
    int16_t start;
    int16_t end;
  };

  struct sparse_tensor_entry_t {
    int32_t offset;
    float probability;
  };

  static constexpr int align(int offset);
};

/*
 * GameLog file format is as follows:
 *
 * [Header]
 * [ValueTensor]                        // game result
 * [pos_index_t...]                     // indices of sampled positions
 * [action_type_t...]                   // MAYBE: skip if only 1 action type?
 * [action_t...]
 * [tensor_index_t...]                  // indices into policy target tensor data
 * [tensor_index_t...]                  // indices into action-value tensor data
 * [Game::State...]                     // all positions, whether sampled or not
 * [sparse_tensor_entry_t...]           // data for sparsely represented tensors (P or AV)
 * [type-0 Tensor...]                   // data for densely represented type-0 tensors (P or AV)
 * [type-1 Tensor...]                   // data for densely represented type-1 tensors (P or AV)
 * ... (as many type-K Tensor sections are there are action types)
 *
 * Each section is aligned to 8 bytes.
 */
template <concepts::Game Game>
class GameLog : public GameLogBase {
 public:
  using Constants = Game::Constants;
  using kNumActionsPerType = Constants::kNumActionsPerType;
  using Rules = Game::Rules;
  using InputTensorizor = Game::InputTensorizor;
  using InputTensor = Game::InputTensorizor::Tensor;
  using TrainingTargetsList = Game::TrainingTargets::List;
  using State = Game::State;
  using Policy = Game::Types::Policy;
  using ActionValues = Game::Types::ActionValues;
  using ValueTensor = Game::Types::ValueTensor;
  using GameLogView = Game::Types::GameLogView;
  using ActionTypeDispatcher = Game::Types::ActionTypeDispatcher;

  static constexpr int kNumActionTypes = Game::Types::kNumActionTypes;

  struct Header {
    uint32_t num_samples = 0;
    uint32_t num_positions = 0;  // includes terminal positions
    uint32_t num_sparse_entries = 0;
    uint32_t num_dense_policies_per_action_type[kNumActionTypes] = {};
  };

  struct MemOffsetTable {
    MemOffsetTable(const Header&);

    int outcome;
    int sampled_indices;
    int action_types;
    int actions;
    int policy_target_indices;
    int action_values_target_indices;
    int states;
    int sparse_tensor_entries;
    int dense_tensors[kNumActionTypes];
  };

  GameLog(const char* filename);
  ~GameLog();

  static ShapeInfo* get_shape_info_array();

  void load(int index, bool apply_symmetry, float* input_values, int* target_indices,
            float** target_arrays) const;

  void replay() const;
  int num_sampled_positions() const;

 private:
  char* get_buffer() const;

  Header& header();
  const Header& header() const;

  int num_positions() const;
  int num_non_terminal_positions() const;

  const pos_index_t* sampled_indices_start_ptr() const;
  const action_type_t* action_type_start_ptr() const;
  const action_t* action_start_ptr() const;
  const tensor_index_t* policy_target_index_start_ptr() const;
  const tensor_index_t* action_values_target_index_start_ptr() const;
  const State* state_start_ptr() const;
  const sparse_tensor_entry_t* sparse_tensor_entry_start_ptr() const;
  template<action_type_t ActionType> const auto* dense_tensor_start_ptr() const;

  Policy get_policy(int state_index) const;
  ActionValues get_action_values(int state_index) const;

  action_type_t get_action_type(int state_index) const;
  const State* get_state(int state_index) const;
  action_t get_prev_action(int state_index) const;
  ValueTensor get_outcome() const;
  pos_index_t get_pos_index(int sample_index) const;

  const std::string filename_;
  char* const buffer_ = nullptr;

  const MemOffsetTable mem_offsets_;
};

template <concepts::Game Game>
class GameLogWriter {
 public:
  using Constants = Game::Constants;
  using kNumActionsPerType = Constants::kNumActionsPerType;
  using Rules = Game::Rules;
  using State = Game::State;
  using ValueTensor = Game::Types::ValueTensor;
  using Policy = Game::Types::Policy;
  using ActionValues = Game::Types::ActionValues;
  using ActionTypeDispatcher = Game::Types::ActionTypeDispatcher;
  using tensor_index_t = GameLogBase::tensor_index_t;
  using sparse_tensor_entry_t = GameLogBase::sparse_tensor_entry_t;
  using tensor_vector_tuple_t =
      mp::Transform_t<std::tuple, Policy, util::make_vector_t>;

  static constexpr int kNumActionTypes = Game::Types::kNumActionTypes;

  struct Entry {
    State position;
    Policy policy_target;
    ActionValues action_values_target;
    action_type_t action_type;
    action_t action;
    bool use_for_training;
    bool policy_target_is_valid;
    bool action_values_target_is_valid;
    bool terminal;
  };
  using entry_vector_t = std::vector<Entry*>;

  GameLogWriter(game_id_t id, int64_t start_timestamp);
  ~GameLogWriter();

  void add(const State& state, action_type_t action_type, action_t action,
           const Policy* policy_target, const ActionValues* action_values,
           bool use_for_training);
  void add_terminal(const State& state, const ValueTensor& outcome);
  void serialize(std::ostream&) const;
  bool was_previous_entry_used_for_policy_training() const;
  int sample_count() const { return sample_count_; }
  game_id_t id() const { return id_; }
  int64_t start_timestamp() const { return start_timestamp_; }

 private:
  template<typename T>
  static void write_section(std::ostream& os, const T* t, int count=1);

  static tensor_index_t write_target(action_type_t, const Policy& target,
                                     tensor_vector_tuple_t& dense_tensors,
                                     std::vector<sparse_tensor_entry_t>& sparse_tensor_entries);

  entry_vector_t entries_;
  ValueTensor outcome_;
  const game_id_t id_;
  const int64_t start_timestamp_;
  int sample_count_ = 0;
  bool terminal_added_ = false;
};

}  // namespace core

#include <inline/core/GameLog.inl>
