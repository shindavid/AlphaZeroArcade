#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/Game.hpp"

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
 * fields are variable-sized. It consists of the following sections, each aligned to 8 bytes:
 *
 *   [State]              // final state
 *   [ValueTensor]        // game result
 *   [pos_index_t...]     // indices of sampled positions
 *   [mem_offset_t...]    // memory-offsets into the DATA region
 *   [DATA]               // differently-sized sections of data
 *
 * The corresponding GameLogMetadata tells us how large each of these sections are.
 *
 * The header tells us how many games are in the file, which allows us to compute the start of the
 * GameData section. Each GameLogMetadata gives us offset information that allows us to seek
 * directly to its corresponding GameData section.
 */
namespace core {

struct ShapeInfo {
  template <eigen_util::concepts::FTensor Tensor>
  void init(const char* nm, int target_idx);
  ~ShapeInfo();

  const char* name = nullptr;
  int* dims = nullptr;
  int num_dims = 0;
  int target_index = -1;
};

struct GameLogFileHeader {
  uint32_t num_games = 0;
  uint32_t num_rows = 0;
  uint64_t reserved = 0;
};
static_assert(sizeof(GameLogFileHeader) == 16);

struct GameLogMetadata {
  uint64_t start_timestamp;
  uint32_t start_offset;  // relative to start of file
  uint32_t data_size;
  uint32_t num_samples;
  uint32_t num_positions;  // excludes terminal position
  uint32_t client_id;
  uint32_t reserved = 0;
};
static_assert(sizeof(GameLogMetadata) == 32);

class GameLogFileReader {
 public:
  GameLogFileReader(const char* buffer);

  const GameLogFileHeader& header() const;
  int num_games() const;

  int num_samples(int game_index) const;
  const char* game_data_buffer(int game_index) const;
  const GameLogMetadata& metadata(int game_index) const;
  const char* buffer() const { return buffer_; }

 private:
  const char* buffer_;
};

struct GameLogCommon {
  static constexpr int kAlignment = 16;

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

  template <typename T>
  static int write_section(std::vector<char>& buf, const T* t, int count = 1, bool pad = true);
};

template <concepts::Game Game>
struct GameLogBase : public GameLogCommon {
  using State = Game::State;
  using PolicyTensor = Game::Types::PolicyTensor;

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
};

template <concepts::Game Game>
class GameReadLog : public GameLogBase<Game> {
 public:
  using mem_offset_t = GameLogCommon::mem_offset_t;
  using pos_index_t = GameLogCommon::pos_index_t;

  using GameLogBase = core::GameLogBase<Game>;
  using Record = GameLogBase::Record;
  using TensorData = GameLogBase::TensorData;

  using Rules = Game::Rules;
  using InputTensorizor = Game::InputTensorizor;
  using InputTensor = Game::InputTensorizor::Tensor;
  using TrainingTargetsList = Game::TrainingTargets::List;
  using State = Game::State;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using ValueTensor = Game::Types::ValueTensor;
  using GameLogView = Game::Types::GameLogView;

  // indicates offsets relative to the start of the GameData region
  struct DataLayout {
    DataLayout(const GameLogMetadata&);

    int final_state;
    int outcome;
    int sampled_indices_start;
    int mem_offsets_start;
    int records_start;
  };

  GameReadLog(const char* filename, int game_index, const GameLogMetadata& metadata,
              const char* buffer);

  static ShapeInfo* get_shape_info_array();

  static void merge_files(const char** input_filenames, int n_input_filenames,
                          const char* output_filename);

  void load(int row_index, bool apply_symmetry, const std::vector<int>& target_indices,
            float* output_array) const;

  // void replay() const;
  int num_sampled_positions() const { return metadata_.num_samples; }

 private:
  static constexpr int align(int offset) { return GameLogCommon::align(offset); }

  int num_positions() const { return metadata_.num_positions; }

  // Populates passed-in tensor and returns true iff a valid tensor is available.
  bool get_policy(mem_offset_t mem_offset, PolicyTensor&) const;

  // Populates passed-in tensor and returns true iff a valid tensor is available.
  bool get_action_values(mem_offset_t mem_offset, ActionValueTensor&) const;

  const State& get_final_state() const;
  const ValueTensor& get_outcome() const;
  pos_index_t get_pos_index(int sample_index) const;
  const Record& get_record(mem_offset_t mem_offset) const;
  mem_offset_t get_mem_offset(int state_index) const;

  const char* filename_;
  const int game_index_;
  const GameLogMetadata& metadata_;
  const char* buffer_ = nullptr;
  const DataLayout layout_;
};

template <concepts::Game Game>
class GameLogSerializer;  // Forward declaration

template <concepts::Game Game>
class GameWriteLog : public GameLogBase<Game> {
 public:
  friend class GameLogSerializer<Game>;
  using mem_offset_t = GameLogCommon::mem_offset_t;
  using pos_index_t = GameLogCommon::pos_index_t;

  using GameLogBase = core::GameLogBase<Game>;
  using Record = GameLogBase::Record;
  using TensorData = GameLogBase::TensorData;

  using Rules = Game::Rules;
  using State = Game::State;
  using ValueTensor = Game::Types::ValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;

  struct Entry {
    State position;
    PolicyTensor policy_target;       // only valid if policy_target_is_valid
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
  bool was_previous_entry_used_for_policy_training() const;
  int sample_count() const { return sample_count_; }
  game_id_t id() const { return id_; }
  int64_t start_timestamp() const { return start_timestamp_; }

 private:
  entry_vector_t entries_;
  State final_state_;
  ValueTensor outcome_;
  const game_id_t id_;
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
template <concepts::Game Game>
class GameLogSerializer {
 public:
  using pos_index_t = GameLogCommon::pos_index_t;
  using mem_offset_t = GameLogCommon::mem_offset_t;
  using GameReadLog = core::GameReadLog<Game>;
  using GameWriteLog = core::GameWriteLog<Game>;

  using Record = GameReadLog::Record;
  using TensorData = GameReadLog::TensorData;
  using Entry = GameWriteLog::Entry;

  GameLogMetadata serialize(const GameWriteLog* log, std::vector<char>& buf, int client_id);

 private:
  std::vector<pos_index_t> sampled_indices_;
  std::vector<mem_offset_t> mem_offsets_;
  std::vector<char> data_buf_;
};

}  // namespace core

#include "inline/core/GameLog.inl"
