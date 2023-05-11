#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/TensorizorConcept.hpp>
#include <util/BoostUtil.hpp>


namespace common {

/*
 * A single TrainingDataWriter is intended to be shared by multiple MctsPlayer's playing in parallel.
 */
template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
class TrainingDataWriter {
public:
  struct Params {
    auto make_options_description();
    bool operator==(const Params& other) const = default;

    std::string games_dir = "c4_games";
    bool clear_dir = true;  // before writing, clear the directory if it exists
  };

  using GameState = GameState_;
  using GameStateTypes = typename common::GameStateTypes<GameState>;
  using Tensorizor = Tensorizor_;
  using TensorizorTypes = typename common::TensorizorTypes<Tensorizor>;

  using GameOutcome = typename GameStateTypes::GameOutcome;

  using InputShape = typename TensorizorTypes::InputShape;
  using PolicyShape = typename GameStateTypes::PolicyShape;
  using ValueShape = typename GameStateTypes::ValueShape;

  using InputTensor = typename TensorizorTypes::InputTensor;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueTensor = typename GameStateTypes::ValueTensor;

  using InputEigenTensor = typename InputTensor::EigenType;
  using PolicyEigenTensor = typename PolicyTensor::EigenType;
  using ValueEigenTensor = typename ValueTensor::EigenType;

  static constexpr size_t InputScalarSize = sizeof(typename InputTensor::Scalar);
  static constexpr size_t PolicyScalarSize = sizeof(typename PolicyTensor::Scalar);
  static constexpr size_t kBytesPerInputRow = InputTensor::Sizes::total_size * InputScalarSize;
  static constexpr size_t kBytesPerPolicyRow = PolicyTensor::Sizes::total_size * PolicyScalarSize;
  static constexpr size_t kEigenStackAllocationLimit = EIGEN_STACK_ALLOCATION_LIMIT;
  static constexpr int kRowsPerChunk = kEigenStackAllocationLimit / std::max(kBytesPerInputRow, kBytesPerPolicyRow);

  using InputChunk = typename TensorizorTypes::template InputTensorN<kRowsPerChunk>;
  using PolicyChunk = typename GameStateTypes::template PolicyTensorN<kRowsPerChunk>;
  using ValueChunk = typename GameStateTypes::template ValueTensorN<kRowsPerChunk>;
  using CurrentPlayerChunk = Eigen::Array<seat_index_t, kRowsPerChunk, 1>;

  using InputBlob = typename TensorizorTypes::DynamicInputTensor;
  using PolicyBlob = typename GameStateTypes::DynamicPolicyTensor;
  using ValueBlob = typename GameStateTypes::DynamicValueTensor;

  struct TensorRefGroup {
    InputEigenTensor& input;
    PolicyEigenTensor& policy;
    ValueEigenTensor& value;
    seat_index_t& current_player;
  };

  /*
   * A single game is recorded onto one or more DataChunk's.
   */
  class DataChunk {
  public:
    DataChunk();

    TensorRefGroup get_next_group();
    void record_for_all(const GameOutcome& value);
    const InputChunk& input() const { return input_; }
    const PolicyChunk& policy() const { return policy_; }
    const ValueChunk& value() const { return value_; }
    int rows() const { return rows_; }
    bool full() const { return rows_ >= kRowsPerChunk; }

  private:
    InputChunk input_;
    PolicyChunk policy_;
    ValueChunk value_;
    CurrentPlayerChunk current_player_;

    int rows_ = 0;
  };

  using data_chunk_list_t = std::list<DataChunk>;

  /*
   * Note: we assume that all GameData read/write operations are from thread-safe contexts. That is, if two self-play
   * Player's hold onto the same GameData, they should read/write to it in separate threads.
   *
   * We can relax this assumption easily by adding some mutexing to this class. With the current overall architecture,
   * that is not necessary.
   */
  class GameData {
  public:
    TensorRefGroup get_next_group();
    void record_for_all(const GameOutcome& value);

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
    data_chunk_list_t chunks_;
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

  static auto input_shape(int rows);
  static auto policy_shape(int rows);
  static auto value_shape(int rows);

  const Params params_;
  std::thread* thread_;
  game_data_map_t game_data_map_;
  game_queue_t game_queue_[2];
  int queue_index_ = 0;
  bool closed_ = false;

  std::condition_variable cv_;
  std::mutex mutex_;

  static TrainingDataWriter* instance_;
};

}  // namespace common

#include <common/inl/TrainingDataWriter.inl>
