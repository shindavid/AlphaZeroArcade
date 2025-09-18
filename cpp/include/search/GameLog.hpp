#pragma once

#include "core/BasicTypes.hpp"
#include "core/InputTensorizor.hpp"
#include "search/AlgorithmsFor.hpp"
#include "search/GameLogBase.hpp"
#include "search/GameLogViewParams.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/MetaProgramming.hpp"

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
namespace search {

template <search::concepts::Traits Traits>
class GameReadLog : public GameLogBase<Traits> {
 public:
  using Game = Traits::Game;
  using GameLogView = Traits::GameLogView;
  using EvalSpec = Traits::EvalSpec;
  using PrimaryTargets = EvalSpec::TrainingTargets::PrimaryList;
  using AuxTargets = EvalSpec::TrainingTargets::AuxList;
  using AllTargets = mp::Concat_t<PrimaryTargets, AuxTargets>;
  using Algorithms = search::AlgorithmsForT<Traits>;

  using mem_offset_t = GameLogCommon::mem_offset_t;
  using pos_index_t = GameLogCommon::pos_index_t;

  using GameLogViewParams = search::GameLogViewParams<Traits>;
  using GameLogBase = search::GameLogBase<Traits>;
  using GameLogCompactRecord = GameLogBase::GameLogCompactRecord;
  using TensorData = GameLogBase::TensorData;

  using Rules = Game::Rules;
  using InputTensorizor = core::InputTensorizor<Game>;
  using InputTensor = InputTensorizor::Tensor;
  using State = Game::State;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using ValueTensor = Game::Types::ValueTensor;

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

  // Populates passed-in tensor and returns true iff a valid tensor is available.
  bool get_action_value_uncertainties(mem_offset_t mem_offset, ActionValueTensor&) const;

  const State& get_final_state() const;
  const ValueTensor& get_outcome() const;
  pos_index_t get_pos_index(int sample_index) const;
  const GameLogCompactRecord& get_record(mem_offset_t mem_offset) const;
  mem_offset_t get_mem_offset(int state_index) const;

  const char* filename_;
  const int game_index_;
  const GameLogMetadata& metadata_;
  const char* buffer_ = nullptr;
  const DataLayout layout_;
};

template <search::concepts::Traits Traits>
class GameLogSerializer;  // Forward declaration

template <search::concepts::Traits Traits>
class GameWriteLog : public GameLogBase<Traits> {
 public:
  friend class GameLogSerializer<Traits>;
  using mem_offset_t = GameLogCommon::mem_offset_t;
  using pos_index_t = GameLogCommon::pos_index_t;

  using GameLogBase = search::GameLogBase<Traits>;
  using GameLogCompactRecord = GameLogBase::GameLogCompactRecord;
  using TensorData = GameLogBase::TensorData;
  using GameLogFullRecord = GameLogBase::GameLogFullRecord;
  using full_record_vec_t = GameLogBase::full_record_vec_t;

  using Game = Traits::Game;
  using TrainingInfo = Traits::TrainingInfo;
  using Rules = Game::Rules;
  using State = Game::State;
  using ValueTensor = Game::Types::ValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using Algorithms = search::AlgorithmsForT<Traits>;

  GameWriteLog(core::game_id_t id, int64_t start_timestamp);
  ~GameWriteLog();

  void add(const TrainingInfo&);

  void add_terminal(const State& state, const ValueTensor& outcome);
  bool was_previous_entry_used_for_policy_training() const;
  int sample_count() const { return sample_count_; }
  core::game_id_t id() const { return id_; }
  int64_t start_timestamp() const { return start_timestamp_; }

 private:
  full_record_vec_t full_records_;
  State final_state_;
  ValueTensor outcome_;
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
template <search::concepts::Traits Traits>
class GameLogSerializer {
 public:
  using Game = Traits::Game;
  using pos_index_t = GameLogCommon::pos_index_t;
  using mem_offset_t = GameLogCommon::mem_offset_t;
  using GameLogBase = search::GameLogBase<Traits>;
  using GameWriteLog = search::GameWriteLog<Traits>;
  using Algorithms = search::AlgorithmsForT<Traits>;

  using GameLogCompactRecord = GameLogBase::GameLogCompactRecord;
  using TensorData = GameLogBase::TensorData;
  using GameLogFullRecord = GameLogBase::GameLogFullRecord;

  GameLogMetadata serialize(const GameWriteLog* log, std::vector<char>& buf, int client_id);

 private:
  std::vector<pos_index_t> sampled_indices_;
  std::vector<mem_offset_t> mem_offsets_;
  std::vector<char> data_buf_;
};

}  // namespace search

#include "inline/search/GameLog.inl"
