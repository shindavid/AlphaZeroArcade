#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

#include <boost/filesystem.hpp>

#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/TensorizorConcept.hpp>

namespace common {

/*
 * A single TrainingDataWriter is intended to be shared by multiple MctsPlayer's playing in parallel.
 */
template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
class TrainingDataWriter {
public:
  using GameState = GameState_;
  using GameStateTypes = typename common::GameStateTypes<GameState>;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using ValueSlab = typename GameStateTypes::ValueSlab;
  using PolicySlab = typename GameStateTypes::PolicySlab;
  using ValueEigenSlab = typename GameStateTypes::ValueEigenSlab;
  using PolicyEigenSlab = typename GameStateTypes::PolicyEigenSlab;

  using Tensorizor = Tensorizor_;
  using TensorizorTypes = typename common::TensorizorTypes<Tensorizor>;
  using TensorShape = typename Tensorizor::Shape;
  using InputSlab = typename TensorizorTypes::InputSlab;
  using InputEigenSlab = typename TensorizorTypes::InputEigenSlab;

  static constexpr size_t kBytesPerInputRow = eigen_util::total_size_v<typename InputSlab::Sizes>
      * sizeof(typename InputSlab::Scalar);
  static constexpr size_t kBytesPerPolicyRow = PolicySlab::Cols * sizeof(typename PolicySlab::Scalar);
  static constexpr size_t kBytesPerValueRow = ValueSlab::Cols * sizeof(typename ValueSlab::Scalar);
  static constexpr size_t kBytesPerRow = kBytesPerInputRow + kBytesPerPolicyRow + kBytesPerValueRow;

  /*
   * For AlphaGo, we have 19 x 19 x 17 input, which with 4-byte floats comes out to 24KB.
   *
   * With 1MB chunks, each chunk can fit about 40 positions. Seems pretty reasonable.
   */
  static constexpr size_t kBytesPerChunk = 1024 * 1024;  // 1MB
  static constexpr size_t kBytesPerFile = 16 * kBytesPerChunk;  // 16MB

  static constexpr int kRowsPerChunk = 1 + kBytesPerChunk / kBytesPerRow;
  static constexpr int kRowsPerFile = 1 + kBytesPerFile / kBytesPerRow;

  static_assert(kRowsPerChunk > 20, "Unreasonably small chunks");

  using InputChunk = typename TensorizorTypes::template InputTensor<kRowsPerChunk>;
  using PolicyChunk = typename GameStateTypes::template PolicyArray<kRowsPerChunk>;
  using ValueChunk = typename GameStateTypes::template ValueArray<kRowsPerChunk>;

  using InputBlob = typename TensorizorTypes::DynamicInputTensor;
  using PolicyBlob = typename GameStateTypes::template PolicyArray<Eigen::Dynamic>;
  using ValueBlob = typename GameStateTypes::template ValueArray<Eigen::Dynamic>;

  struct EigenSlab {
    InputEigenSlab& input;
    PolicyEigenSlab& policy;
    ValueEigenSlab& value;
  };

  /*
   * A single game is recorded onto one or more DataChunk's.
   */
  class DataChunk {
  public:
    DataChunk();

    EigenSlab get_next_slab();
    void record_for_all(const ValueEigenSlab& value);
    const InputChunk& input() const { return input_; }
    const PolicyChunk& policy() const { return policy_; }
    const ValueChunk& value() const { return value_; }
    int rows() const { return rows_; }
    bool full() const { return rows_ >= kRowsPerChunk; }

  private:
    InputChunk input_;
    PolicyChunk policy_;
    ValueChunk value_;

    int rows_ = 0;
  };

  using data_chunk_list_t = std::list<DataChunk>;

  class GameData {
  public:
    EigenSlab get_next_slab();
    void record_for_all(const ValueEigenSlab& value);
    const data_chunk_list_t& chunks() const { return chunks_; }

  private:
    DataChunk* get_next_chunk();
    data_chunk_list_t chunks_;
  };

  TrainingDataWriter(const boost::filesystem::path& output_path);
  ~TrainingDataWriter();

  GameData* allocate_data() { return new GameData(); }

  /*
   * Takes ownership of pointer.
   */
  void process(const GameData* data);

  void close();

protected:
  using game_queue_t = std::vector<const GameData*>;

  void flush();
  void loop();
  void write(const DataChunk& chunk);
  void partial_write(const DataChunk& chunk, int start, int end);

  static auto input_shape(int rows);
  static auto policy_shape(int rows);
  static auto value_shape(int rows);

  const boost::filesystem::path output_path_;
  std::thread* thread_;
  game_queue_t game_queue_[2];
  int queue_index_ = 0;
  bool closed_ = false;

  InputBlob input_;
  PolicyBlob policy_;
  ValueBlob value_;

  std::condition_variable cv_;
  std::mutex mutex_;
  int file_number_ = 0;
  int rows_ = 0;
};

}  // namespace common

#include <common/inl/TrainingDataWriter.inl>
