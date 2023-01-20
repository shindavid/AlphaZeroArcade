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

namespace common {

/*
 * A single TrainingDataWriter is intended to be shared by multiple MctsPlayer's playing in parallel.
 */
template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
class TrainingDataWriter {
public:
  struct Params {
    std::string games_dir = "c4_games";
    bool clear_dir = true;  // before writing, clear the directory if it exists

    void add_options(boost::program_options::options_description& desc, bool add_shortcuts=false);
  };

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
  static constexpr int kRowsPerChunk = 1 + kBytesPerChunk / kBytesPerRow;
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

  /*
   * Note: we assume that all GameData read/write operations are from thread-safe contexts. That is, if two self-play
   * Player's hold onto the same GameData, they should read/write to it in separate threads.
   *
   * We can relax this assumption easily by adding some mutexing to this class. With the current overall architecture,
   * that is not necessary.
   */
  class GameData {
  public:
    EigenSlab get_next_slab();
    void record_for_all(const ValueEigenSlab& value);

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

  TrainingDataWriter(const Params& params);
  ~TrainingDataWriter();

  GameData_sptr get_data(game_id_t id);

  void close(GameData_sptr data);

  void shut_down();

protected:
  using game_queue_t = std::vector<GameData_sptr>;

  void loop();
  void write_to_file(const GameData* data);

  static auto input_shape(int rows);
  static auto policy_shape(int rows);
  static auto value_shape(int rows);

  const boost::filesystem::path output_path_;
  std::thread* thread_;
  game_data_map_t game_data_map_;
  game_queue_t game_queue_[2];
  int queue_index_ = 0;
  bool closed_ = false;

  std::condition_variable cv_;
  std::mutex mutex_;
};

}  // namespace common

#include <common/inl/TrainingDataWriter.inl>
