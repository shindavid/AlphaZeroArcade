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

  struct Header {
    uint32_t num_samples = 0;
    uint32_t num_positions = 0;  // includes terminal positions
    uint32_t num_dense_policies = 0;
    uint32_t num_sparse_policy_entries = 0;
    uint32_t num_dense_action_values = 0;
    uint32_t num_sparse_action_values_entries = 0;
    uint64_t extra = 0;  // leave extra space for future use (version numbering, etc.)
  };
  static_assert(sizeof(Header) == 32);

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
 * [action_t...]
 * [tensor_index_t...]                  // indices into policy target tensor data
 * [tensor_index_t...]                  // indices into action-value tensor data
 * [Game::State...]                     // all positions, whether sampled or not
 * [Game::Types::PolicyTensor...]       // data for densely represented policy targets
 * [sparse_tensor_entry_t...]           // data for sparsely represented policy targets
 * [Game::Types::ActionValueTensor...]  // data for densely represented action value targets
 * [sparse_tensor_entry_t...]           // data for sparsely represented action value targets
 *
 * Each section is aligned to 8 bytes.
 */
template <concepts::Game Game>
class GameLog : public GameLogBase {
 public:
  using Header = GameLogBase::Header;

  using Rules = Game::Rules;
  using InputTensorizor = Game::InputTensorizor;
  using InputTensor = Game::InputTensorizor::Tensor;
  using TrainingTargetsList = Game::TrainingTargets::List;
  using State = Game::State;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using ValueTensor = Game::Types::ValueTensor;
  using GameLogView = Game::Types::GameLogView;

  GameLog(const char* filename);
  ~GameLog();

  static ShapeInfo* get_shape_info_array();

  void load(int index, bool apply_symmetry, float* input_values, int* target_indices,
            float** target_arrays, bool** target_masks) const;

  void replay() const;
  int num_sampled_positions() const;

 private:
  char* get_buffer() const;
  Header& header();
  const Header& header() const;

  int num_positions() const;
  int num_non_terminal_positions() const;
  int num_dense_policies() const;
  int num_sparse_policy_entries() const;
  int num_dense_action_values() const;
  int num_sparse_action_values_entries() const;

  static constexpr int header_start_mem_offset();
  static constexpr int outcome_start_mem_offset();
  static constexpr int sampled_indices_start_mem_offset();
  int action_start_mem_offset() const;
  int policy_target_index_start_mem_offset() const;
  int action_values_target_index_start_mem_offset() const;
  int state_start_mem_offset() const;
  int dense_policy_start_mem_offset() const;
  int sparse_policy_entry_start_mem_offset() const;
  int dense_action_values_start_mem_offset() const;
  int sparse_action_values_entry_start_mem_offset() const;

  const action_t* action_start_ptr() const;
  const tensor_index_t* policy_target_index_start_ptr() const;
  const tensor_index_t* action_values_target_index_start_ptr() const;
  const State* state_start_ptr() const;
  const PolicyTensor* dense_policy_start_ptr() const;
  const sparse_tensor_entry_t* sparse_policy_entry_start_ptr() const;
  const ActionValueTensor* dense_action_values_start_ptr() const;
  const sparse_tensor_entry_t* sparse_action_values_entry_start_ptr() const;
  const pos_index_t* sampled_indices_start_ptr() const;

  bool get_policy(int state_index, PolicyTensor&) const;  // return true iff valid
  bool get_action_values(int state_index, ActionValueTensor&) const;  // return true iff valid
  const State* get_state(int state_index) const;
  action_t get_prev_action(int state_index) const;
  ValueTensor get_outcome() const;
  pos_index_t get_pos_index(int sample_index) const;

  const std::string filename_;
  char* const buffer_ = nullptr;
  const int action_start_mem_offset_;
  const int policy_target_index_start_mem_offset_;
  const int action_values_target_index_start_mem_offset_;
  const int state_start_mem_offset_;
  const int dense_policy_start_mem_offset_;
  const int sparse_policy_entry_start_mem_offset_;
  const int dense_action_values_start_mem_offset_;
  const int sparse_action_values_entry_start_mem_offset_;
};

template <concepts::Game Game>
class GameLogWriter {
 public:
  using Rules = Game::Rules;
  using State = Game::State;
  using ValueTensor = Game::Types::ValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using tensor_index_t = GameLogBase::tensor_index_t;
  using sparse_tensor_entry_t = GameLogBase::sparse_tensor_entry_t;

  struct Entry {
    State position;
    PolicyTensor policy_target;  // only valid if policy_target_is_valid
    ActionValueTensor action_values;  // only valid if action_values_are_valid
    action_t action;
    bool use_for_training;
    bool policy_target_is_valid;
    bool action_values_are_valid;
    bool terminal;
  };
  using entry_vector_t = std::vector<Entry*>;

  GameLogWriter(game_id_t id, int64_t start_timestamp);
  ~GameLogWriter();

  void add(const State& state, action_t action, const PolicyTensor* policy_target,
           const ActionValueTensor* action_values, bool use_for_training);
  void add_terminal(const State& state, const ValueTensor& outcome);
  void serialize(std::ostream&) const;
  bool was_previous_entry_used_for_policy_training() const;
  int sample_count() const { return sample_count_; }
  game_id_t id() const { return id_; }
  int64_t start_timestamp() const { return start_timestamp_; }

 private:
  template<typename T>
  static void write_section(std::ostream& os, const T* t, int count=1);

  template <eigen_util::concepts::FTensor Tensor>
  static tensor_index_t write_target(
      const Tensor& target, std::vector<Tensor>& dense_tensors,
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
