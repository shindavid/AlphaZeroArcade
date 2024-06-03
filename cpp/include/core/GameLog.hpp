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

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
class GameWriteLog {
 public:
  using Action = typename GameState::Action;
  using GameStateData = typename GameState::Data;

  using GameStateTypes = core::GameStateTypes<GameState>;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using SymmetryIndexSet = GameStateTypes::SymmetryIndexSet;

  struct Entry {
    GameStateData state_data;
    SymmetryIndexSet symmetries;
    PolicyTensor policy_target;  // only valid if policy_target_is_valid
    Action action;
    bool use_for_training;
    bool policy_target_is_valid;
    bool terminal;
  };

  using entry_vector_t = std::vector<Entry*>;

  GameWriteLog(game_id_t id, int64_t start_timestamp);
  ~GameWriteLog();

  void add(const GameState& state, const Action& action, const PolicyTensor* policy_target,
           bool use_for_training);
  void add_terminal(const GameState& state, const GameOutcome& outcome);
  void serialize(std::ostream&) const;
  bool is_previous_entry_used_for_training() const;
  int sym_train_count() const { return sym_train_count_; }
  game_id_t id() const { return id_; }
  int64_t start_timestamp() const { return start_timestamp_; }
  void close() { closed_ = true; }
  bool closed() const { return closed_; }

 private:
  // If policy target is not valid, returns 0.
  // Else, decides whether to write a sparse or dense representation based on the number of non-zero
  // entries. If writing a dense representation, returns -1. Else, returns the number of entries
  // written.
  static int16_t write_policy_target(const Entry& entry, char** buffer);

  entry_vector_t entries_;
  GameOutcome outcome_ = GameStateTypes::make_non_terminal_outcome();
  const game_id_t id_;
  const int64_t start_timestamp_;
  int sym_train_count_ = 0;
  bool terminal_added_ = false;
  bool closed_ = false;
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
template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
class GameReadLog {
 public:
  using Action = typename GameState::Action;
  using GameStateData = typename GameState::Data;

  using InputTensor = typename Tensorizor::InputTensor;
  using AuxTargetList = typename Tensorizor::AuxTargetList;

  using GameStateTypes = core::GameStateTypes<GameState>;
  using TensorizorTypes = core::TensorizorTypes<Tensorizor>;

  using GameOutcome = typename GameStateTypes::GameOutcome;
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

  /*
   * Returns the index'th dimension of the tensor with the given key.
   *
   * If index is out of bounds, returns -1.
   *
   * If key is not recognized, returns -2.
   */
  static int get_dim(const char* key, int index);

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
  GameOutcome outcome_;
};

}  // namespace core

#include <inline/core/GameLog.inl>
