#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <core/AbstractSymmetryTransform.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <util/BoostUtil.hpp>

namespace core {

/*
 * A single TrainingDataWriter is intended to be shared by multiple MctsPlayer's playing in
 * parallel.
 */
template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
class TrainingDataWriter {
 public:
  struct Params {
    auto make_options_description();
    bool operator==(const Params& other) const = default;

    std::string games_dir = "c4_games";
    int64_t max_rows = 0;
  };

  using GameState = GameState_;
  using GameStateTypes = typename core::GameStateTypes<GameState>;
  using Tensorizor = Tensorizor_;
  using TensorizorTypes = typename core::TensorizorTypes<Tensorizor>;

  using GameOutcome = typename GameStateTypes::GameOutcome;
  using AuxTargetList = typename TensorizorTypes::AuxTargetList;
  using AuxTargetTensorTuple = typename TensorizorTypes::AuxTargetTensorTuple;
  using AuxTargetTorchTensorTuple = typename TensorizorTypes::AuxTargetTorchTensorTuple;

  using InputShape = typename TensorizorTypes::InputShape;
  using PolicyShape = typename GameStateTypes::PolicyShape;
  using ValueShape = typename GameStateTypes::ValueShape;

  using InputTensor = typename TensorizorTypes::InputTensor;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueTensor = typename GameStateTypes::ValueTensor;

  using InputScalar = torch_util::convert_type_t<typename InputTensor::Scalar>;
  using PolicyScalar = torch_util::convert_type_t<typename PolicyTensor::Scalar>;
  using ValueScalar = torch_util::convert_type_t<typename ValueTensor::Scalar>;

  using PolicyTransform = AbstractSymmetryTransform<PolicyTensor>;

  static constexpr int kRowsPerChunk = 64;
  static constexpr int kNumAuxTargets = mp::Length_v<AuxTargetList>;

  struct TensorGroup {
    GameState state;
    InputTensor input;
    PolicyTensor policy;
    PolicyTensor opp_policy;
    ValueTensor value;
    AuxTargetTensorTuple aux_targets;
    seat_index_t current_player;
    symmetry_index_t sym_index;
  };
  using group_vec_t = std::vector<TensorGroup*>;

  /*
   * A single game is recorded onto one or more DataChunk's.
   */
  class DataChunk {
   public:
    DataChunk();
    ~DataChunk();

    TensorGroup& get_next_group();
    void record_for_all(const GameState& state, const GameOutcome& value);
    int rows() const { return rows_; }
    bool full() const { return rows_ >= kRowsPerChunk; }

    const TensorGroup& get_group(int i) const { return tensors_[i]; }

   private:
    TensorGroup* tensors_;
    int rows_ = 0;
  };

  using data_chunk_list_t = std::list<DataChunk>;

  /*
   * Note: we assume that all GameData read/write operations are from thread-safe contexts. That is,
   * if two self-play Player's hold onto the same GameData, they should read/write to it in separate
   * threads.
   *
   * We can relax this assumption easily by adding some mutexing to this class. With the current
   * overall architecture, that is not necessary.
   */
  class GameData {
   public:
    TensorGroup& get_next_group();
    void record_for_all(const GameState& state, const GameOutcome& value);
    void add_pending_group(TensorGroup* group) { pending_groups_.push_back(group); }
    bool contains_pending_groups() const { return !pending_groups_.empty(); }
    void commit_opp_reply_to_pending_groups(const PolicyTensor& opp_policy);

   protected:
    // friend classes only intended to use these protected members
    GameData(game_id_t id) : id_(id) {}
    const data_chunk_list_t& chunks() const { return chunks_; }
    game_id_t id() const { return id_; }
    bool closed() const { return closed_; }
    void close() { closed_ = true; }

    friend class TrainingDataWriter;

   private:
    // friend classes not intended to use these private members
    DataChunk* get_next_chunk();

    std::mutex mutex_;
    data_chunk_list_t chunks_;
    group_vec_t pending_groups_;  // for opponent-reply auxiliary policy target
    const game_id_t id_;
    bool closed_ = false;
  };
  using GameData_sptr = std::shared_ptr<GameData>;
  using game_data_map_t = std::map<game_id_t, GameData_sptr>;

  static TrainingDataWriter* instantiate(const Params& params);

  /*
   * Assumes that instantiate() was called at least once.
   */
  static TrainingDataWriter* instance() { return instance_; }

  GameData_sptr get_data(game_id_t id);

  void close(GameData_sptr data);

  void shut_down();

 protected:
  using game_queue_t = std::vector<GameData_sptr>;

  boost::filesystem::path games_dir() const { return params_.games_dir; }
  TrainingDataWriter(const Params& params);
  ~TrainingDataWriter();

  void loop();
  void write_to_file(const GameData* data);

  const Params params_;
  std::thread* thread_;
  game_data_map_t game_data_map_;
  game_queue_t game_queue_[2];
  int64_t rows_written_ = 0;
  int queue_index_ = 0;
  bool closed_ = false;

  std::condition_variable cv_;
  std::mutex mutex_;

  static TrainingDataWriter* instance_;
};

}  // namespace core

#include <inline/core/TrainingDataWriter.inl>
