#pragma once

#include <core/BasicTypes.hpp>
#include <core/Constants.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/Symmetries.hpp>
#include <core/TensorizorConcept.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace core {

struct ShapeInfo {
  template <eigen_util::concepts::FTensor Tensor> void init(const char* name);
  ~ShapeInfo();

  const char* name = nullptr;
  int* dims = nullptr;
  int num_dims = 0;
};

struct GameLogBase {
  struct Header {
    uint32_t num_samples_with_symmetry_expansion = 0;
    uint32_t num_samples_without_symmetry_expansion = 0;
    uint32_t num_positions = 0;  // includes terminal positions
    uint32_t extra = 0;          // leave extra space for future use (version numbering, etc.)
  };

  struct sym_sample_index_t {
    uint32_t pos_index;
    uint32_t sym_index;
  };

#pragma pack(push, 1)
  struct non_sym_sample_index_t {
    uint32_t pos_index;
  };
#pragma pack(pop)

#pragma pack(push, 1)
  struct policy_tensor_index_t {
    // If start < end, then the representation is sparse, and entries [start, end) of
    // the sparse_policy_entry_t region represent the sparse tensor.
    // Else, if start == end >= 0, then start is the index into the dense tensor region.
    // Else, if start == end < 0, then there is no policy target.
    // Any other start/end pair is invalid.
    int16_t start;
    int16_t end;
  };
#pragma pack(pop)

  struct sparse_policy_entry_t {
    int offset;
    float probability;
  };
};

/*
 * GameLog file format is as follows:
 *
 * [Header]
 * [ValueArray]
 * [sym_sample_index_t...]
 * [non_sym_sample_index_t...]
 * [action_t...]
 * [policy_tensor_index_t...]
 * [Game::StateSnapshot...]
 * [Game::PolicyTensor...]  // data for densely represented tensors
 * [sparse_policy_entry_t...]  // data for sparsely represented tensors
 *
 * Each section is aligned to 8 bytes.
 */
template <core::concepts::Game Game>
class GameLog : public GameLogBase {
 public:
  using Header = GameLogBase::Header;
  using ValueArray = typename Game::ValueArray;
  using sym_sample_index_t = GameLogBase::sym_sample_index_t;
  using non_sym_sample_index_t = GameLogBase::non_sym_sample_index_t;
  using policy_tensor_index_t = GameLogBase::policy_tensor_index_t;
  using sparse_policy_entry_t = GameLogBase::sparse_policy_entry_t;

 private:
  int header_start() const;
  int outcome_start() const;
  int sym_sample_index_start() const;
  int non_sym_sample_index_start() const;
  int action_start() const;
  int policy_tensor_index_start() const;
  int position_start() const;
  int dense_policy_start() const;
  int sparse_policy_start() const;

  char* buffer_;
};

template <concepts::Game Game>
class GameLogWriter {
 public:
  using Rules = typename Game::Rules;
  using StateSnapshot = typename Game::StateSnapshot;
  using FullState = typename Game::FullState;
  using ValueArray = typename Game::ValueArray;
  using SymmetryIndexSet = typename Game::SymmetryIndexSet;
  using PolicyTensor = eigen_util::FTensor<typename Game::PolicyShape>;

  struct Entry {
    StateSnapshot position;
    SymmetryIndexSet symmetries;
    PolicyTensor policy_target;  // only valid if policy_target_is_valid
    action_t action;
    bool use_for_training;
    bool policy_target_is_valid;
    bool terminal;
  };
  using entry_vector_t = std::vector<Entry*>;

  GameLogWriter(game_id_t id, int64_t start_timestamp);
  ~GameLogWriter();

  void add(const FullState& state, action_index_t action, const PolicyTensor* policy_target,
           bool use_for_training);
  void add_terminal(const FullState& state, const ValueArray& outcome);
  void serialize(std::ostream&) const;
  bool is_previous_entry_used_for_training() const;
  int sym_train_count() const { return sym_train_count_; }
  game_id_t id() const { return id_; }
  int64_t start_timestamp() const { return start_timestamp_; }
  void close() { closed_ = true; }
  bool closed() const { return closed_; }

 private:
  template<typename T>
  static void write_section(std::ostream& os, const T* t, int count=1);

  static GameLogBase::policy_tensor_index_t write_policy_target(
      const Entry& entry, std::vector<PolicyTensor>& dense_tensors,
      std::vector<GameLogBase::sparse_policy_entry_t>& sparse_tensor_entries);

  entry_vector_t entries_;
  ValueArray outcome_;
  const game_id_t id_;
  const int64_t start_timestamp_;
  int sym_train_count_ = 0;
  int non_sym_train_count_ = 0;
  bool terminal_added_ = false;
  bool closed_ = false;
};

template <concepts::Game Game>
class GameLogReader {
 public:
  // TODO
};

// File format:
//
// [ HEADER ]
// [ GAME_OUTCOME ]
// [ SAMPLING INDEX WITH SYMMETRY EXPANSION ]
// [ SAMPLING INDEX WITHOUT SYMMETRY EXPANSION ]
// [ AUX INDEX ]
// [ STATE DATA ]
// [ AUX DATA ]
//
// The AUX DATA section uses variable-sized entries, as the policy target representation can be
// either sparse or dense.
template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
class GameReadLog {
 public:
  using Action = typename GameState::Action;
  using GameStateData = typename GameState::Data;

  using InputTensor = typename Tensorizor::InputTensor;
  using AuxTargetList = typename Tensorizor::AuxTargetList;

  using GameStateTypes = core::GameStateTypes<GameState>;
  using TensorizorTypes = core::TensorizorTypes<Tensorizor>;

  using ValueArray = typename GameStateTypes::ValueArray;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueTensor = typename GameStateTypes::ValueTensor;
  using GameStateHistory = typename TensorizorTypes::GameStateHistory;

  using Transforms = core::Transforms<GameState>;
  using Transform = typename Transforms::Transform;

  struct sparse_policy_entry_t {
    int offset;
    float probability;
  };

  struct Header {
    uint32_t num_samples_with_symmetry_expansion = 0;
    uint32_t num_samples_without_symmetry_expansion = 0;
    uint32_t num_game_states = 0;  // includes terminal state
    uint32_t extra = 0;  // leave extra space for future use (version numbering, etc.)
  };

  // TODO: consider packing SymSampleIndex to fit in 4 bytes
  struct SymSampleIndex {
    uint32_t state_index;
    uint32_t symmetry_index;
  };

  using NonSymSampleIndex = uint32_t;
  using AuxMemOffset = int32_t;

#pragma pack(push, 1)
  // NOTE: when we change Action to be an int32_t, this will fit in 8 bytes
  struct AuxData {
    Action action;

    // If -1, then representation is dense
    // If 0, then there is no policy target
    // If >0, then representation is sparse. The value is the number of entries.
    int16_t policy_target_format;
    uint16_t use_for_training;
  };
#pragma pack(pop)

  GameReadLog(const char* filename);
  ~GameReadLog();

  void load(int index, bool apply_symmetry, const char** keys, float** values, int num_keys);

  static ShapeInfo* get_shape_info();

 private:
  void load_policy(PolicyTensor* policy, AuxMemOffset aux_mem_offset);

  template<typename T>
  void seek_and_read(int offset, T* data, int count=1);

  int outcome_start() const;
  int symmetric_index_start() const;
  int non_symmetric_index_start() const;
  int aux_index_start() const;
  int state_data_start() const;
  int aux_data_start() const;

  int symmetric_index_offset(int index) const;
  int non_symmetric_index_offset(int index) const;
  int aux_index_offset(int index) const;
  int state_data_offset(int index) const;
  int aux_data_offset(int mem_offset) const;

  uint32_t num_aux_data() const { return header_.num_game_states - 1; }  // no aux for terminal

  std::string filename_;
  FILE* file_;
  Header header_;
  ValueArray outcome_;
};

}  // namespace core

#include <inline/core/GameLog.inl>
