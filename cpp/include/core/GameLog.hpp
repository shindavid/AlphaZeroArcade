#pragma once

#include <core/BasicTypes.hpp>
#include <core/Constants.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace core {

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
class GameWriteLog {
 public:
  using Action = typename GameState::Action;
  using GameStateTypes = core::GameStateTypes<GameState>;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;

  struct Entry {
    GameState state;
    PolicyTensor policy_target;  // only valid if policy_target_is_valid
    Action action;
    core::SearchMode search_mode;
    bool policy_target_is_valid;
  };

  using entry_vector_t = std::vector<Entry*>;

  ~GameWriteLog();

  void add(const GameState& state, const Action& action, const PolicyTensor* policy_target,
           core::SearchMode search_mode);
  void add_terminal(const GameState& state, const GameOutcome& outcome);
  void serialize(std::ostream&) const;

 private:
  // If policy target is not valid, returns 0.
  // Else, decides whether to write a sparse or dense representation based on the number of non-zero
  // entries. If writing a dense representation, returns -1. Else, returns the number of entries
  // written.
  static int write_policy_target(const Entry& entry, char** buffer);

  entry_vector_t entries_;
  GameOutcome outcome_ = GameStateTypes::make_non_terminal_outcome();
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

  using GameStateTypes = core::GameStateTypes<GameState>;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueTensor = typename GameStateTypes::ValueTensor;

  struct sparse_policy_entry_t {
    int offset;
    float probability;
  };

  struct Header {
    uint64_t extra = 0;  // leave extra space for future use (version numbering, etc.)
    uint32_t num_samples_with_symmetry_expansion = 0;
    uint32_t num_samples_without_symmetry_expansion = 0;
    uint32_t num_aux_entries = 0;
    uint32_t num_game_states = 0;
  };

  struct SymSampleIndex {
    uint32_t state_index;
    uint8_t symmetry_index;
  };

  using NonSymSampleIndex = uint32_t;

  struct AuxIndex {
    uint32_t state_index;
    uint32_t aux_mem_offset;
  };

  struct AuxData {
    Action action;
    core::SearchMode search_mode;

    // If -1, then representation is dense
    // If 0, then there is no policy target
    // If >0, then representation is sparse. The value is the number of entries.
    int32_t policy_target_format;
  };

  GameReadLog(const char* filename);
  ~GameReadLog();

  void load(int index, bool apply_symmetry, float* input, float* policy, float* value,
            const char** aux_keys, float* aux);

 private:
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

  std::string filename_;
  FILE* file_;
  Header header_;
  GameOutcome outcome_;
};

}  // namespace core

#include <inline/core/GameLog.inl>
