#pragma once

#include <core/BasicTypes.hpp>
#include <core/Constants.hpp>
#include <core/concepts/Game.hpp>

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
    uint32_t num_positions = 0;  // excludes terminal position
    uint64_t extra = 0;  // leave extra space for future use (version numbering, etc.)
  };
  static_assert(sizeof(Header) == 16);

  using pos_index_t = int32_t;
  using mem_offset_t = int32_t;

  // tensor_encoding_t
  //
  // Used in TensorData. A value of t indicates that the TensorData::data field contains 4*abs(t)
  // bytes of data.
  //
  // A negative value indicates that a dense tensor is stored, and a positive value indicates that a
  // sparse tensor is stored.
  using tensor_encoding_t = int32_t;

  struct SparseTensorEntry {
    int32_t offset;
    float probability;
  };

  static constexpr int align(int offset);
};

/*
 * GameReadLog file format is as follows:
 *
 * [Header]
 * [State]                              // final state
 * [ValueTensor]                        // game result
 * [pos_index_t...]                     // indices of sampled positions
 * [mem_offset_t...]                    // memory-offsets into the DATA region
 * [DATA]                               // differently-sized sections of data
 *
 * Each section is aligned to 8 bytes.
 */
template <concepts::Game Game>
class GameReadLog : public GameLogBase {
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

  struct Record {
    State position;
    seat_index_t active_seat;
    action_mode_t action_mode;
    action_t action;
  };

  struct TensorData {
    static constexpr int kDenseCapacity = PolicyTensor::Dimensions::total_size;
    static constexpr int kSparseCapacity = PolicyTensor::Dimensions::total_size / 2;

    TensorData(bool valid, const PolicyTensor&);
    int write_to(std::vector<char>& buf) const;
    int size() const { return sizeof(encoding) + 4 * std::abs(encoding); }
    bool load(PolicyTensor&) const;  // return true if valid tensor

    struct DenseData {
      float x[kDenseCapacity];
    };

    struct SparseData {
      SparseTensorEntry x[kSparseCapacity];
    };
    static_assert(sizeof(SparseData) == 8 * kSparseCapacity);

    union data_t {
      DenseData dense_repr;
      SparseData sparse_repr;
    };

    tensor_encoding_t encoding;
    data_t data;
  };
  static_assert(sizeof(TensorData) ==
                sizeof(tensor_encoding_t) + sizeof(typename TensorData::data_t));

  struct FileLayout {
    FileLayout(const Header&);

    int header;
    int final_state;
    int outcome;
    int sampled_indices_start;
    int mem_offsets_start;
    int records_start;
  };

  GameReadLog(const char* filename);
  ~GameReadLog();

  static ShapeInfo* get_shape_info_array();

  void load(int index, bool apply_symmetry, float* input_values, int* target_indices,
            float** target_arrays, bool** target_masks) const;

  void replay() const;
  int num_sampled_positions() const { return header().num_samples; }

 private:
  char* get_buffer() const;
  const Header& header() const;

  int num_positions() const { return header().num_positions; }

  // Populates passed-in tensor and returns true iff a valid tensor is available.
  bool get_policy(mem_offset_t mem_offset, PolicyTensor&) const;

  // Populates passed-in tensor and returns true iff a valid tensor is available.
  bool get_action_values(mem_offset_t mem_offset, ActionValueTensor&) const;

  const State& get_final_state() const;
  const ValueTensor& get_outcome() const;
  pos_index_t get_pos_index(int sample_index) const;
  const Record& get_record(mem_offset_t mem_offset) const;
  mem_offset_t get_mem_offset(int state_index) const;

  const std::string filename_;
  char* const buffer_ = nullptr;
  FileLayout layout_;
};

template <concepts::Game Game>
class GameWriteLog : public GameLogBase {
 public:
  using GameReadLog = core::GameReadLog<Game>;
  using Record = GameReadLog::Record;
  using TensorData = GameReadLog::TensorData;

  using Rules = Game::Rules;
  using State = Game::State;
  using ValueTensor = Game::Types::ValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;

  struct Entry {
    State position;
    PolicyTensor policy_target;  // only valid if policy_target_is_valid
    ActionValueTensor action_values;  // only valid if action_values_are_valid
    action_t action;
    seat_index_t active_seat;
    bool use_for_training;
    bool policy_target_is_valid;
    bool action_values_are_valid;
  };
  using entry_vector_t = std::vector<Entry*>;

  GameWriteLog(game_id_t id, int64_t start_timestamp);
  ~GameWriteLog();

  void add(const State& state, action_t action, seat_index_t active_seat,
           const PolicyTensor* policy_target, const ActionValueTensor* action_values,
           bool use_for_training);
  void add_terminal(const State& state, const ValueTensor& outcome);
  void serialize(std::vector<char>& buf) const;
  bool was_previous_entry_used_for_policy_training() const;
  int sample_count() const { return sample_count_; }
  game_id_t id() const { return id_; }
  int64_t start_timestamp() const { return start_timestamp_; }

 private:
  template <typename T>
  static int write_section(std::vector<char>& buf, const T* t, int count=1, bool pad=true);

  entry_vector_t entries_;
  State final_state_;
  ValueTensor outcome_;
  const game_id_t id_;
  const int64_t start_timestamp_;
  int sample_count_ = 0;
  bool terminal_added_ = false;
};

}  // namespace core

#include <inline/core/GameLog.inl>
