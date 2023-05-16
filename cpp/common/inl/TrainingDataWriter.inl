#include <common/TrainingDataWriter.hpp>

#include <filesystem>
#include <map>
#include <string>

#include <util/BoostUtil.hpp>
#include <util/StringUtil.hpp>
#include <util/TorchUtil.hpp>

namespace common {

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>* TrainingDataWriter<GameState_, Tensorizor_>::instance_ = nullptr;

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
auto TrainingDataWriter<GameState_, Tensorizor_>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("TrainingDataWriter options");
  return desc
      .template add_bool_switches<"clear-dir", "no-clear-dir">(
          &clear_dir, "rm {games-dir}/* before running", "do NOT rm {games-dir}/* before running")
      .template add_option<"games-dir", 'g'>(
          po::value<std::string>(&games_dir)->default_value(games_dir.c_str()), "where to write games")
      ;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::DataChunk()
: input_(input_shape(kRowsPerChunk))
, policy_(policy_shape(kRowsPerChunk))
, value_(value_shape(kRowsPerChunk)) {}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
typename TrainingDataWriter<GameState_, Tensorizor_>::TensorRefGroup
TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::get_next_group() {
  int rows = rows_++;
  return TensorRefGroup{
    input_.template eigenSlice<InputShape>(rows),
    policy_.template eigenSlice<PolicyShape>(rows),
    value_.template eigenSlice<ValueShape>(rows),
    current_player_[rows]
    };
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::record_for_all(const GameOutcome& value) {
  for (int i = 0; i < rows_; ++i) {
    GameOutcome shifted_value = value;
    eigen_util::left_rotate(shifted_value, current_player_[i]);
    value_.template eigenSlice<ValueShape>(i) = eigen_util::reinterpret_as_tensor<ValueEigenTensor>(shifted_value);
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
auto TrainingDataWriter<GameState_, Tensorizor_>::TrainingDataWriter::input_shape(int rows) {
  return util::to_std_array<int>(rows, eigen_util::to_int64_std_array_v<InputShape>);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
auto TrainingDataWriter<GameState_, Tensorizor_>::TrainingDataWriter::policy_shape(int rows) {
  return util::to_std_array<int>(rows, eigen_util::to_int64_std_array_v<PolicyShape>);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
auto TrainingDataWriter<GameState_, Tensorizor_>::TrainingDataWriter::value_shape(int rows) {
  return util::to_std_array<int>(rows, GameState::kNumPlayers);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
typename TrainingDataWriter<GameState_, Tensorizor_>::TensorRefGroup
TrainingDataWriter<GameState_, Tensorizor_>::GameData::get_next_group() {
  std::unique_lock<std::mutex> lock(mutex_);
  return get_next_chunk()->get_next_group();
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
typename TrainingDataWriter<GameState_, Tensorizor_>::DataChunk*
TrainingDataWriter<GameState_, Tensorizor_>::GameData::get_next_chunk() {
  if (chunks_.empty() || chunks_.back().full()) {
    chunks_.emplace_back();
  }
  return &chunks_.back();
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::GameData::record_for_all(const GameOutcome& value) {
  std::unique_lock<std::mutex> lock(mutex_);
  for (DataChunk& chunk : chunks_) {
    chunk.record_for_all(value);
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>*
TrainingDataWriter<GameState_, Tensorizor_>::instantiate(const Params& params) {
  if (!instance_) {
    instance_ = new TrainingDataWriter(params);
  } else {
    if (params != instance_->params_) {
      throw std::runtime_error("TrainingDataWriter::instance() called with different params");
    }
  }
  return instance_;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
typename TrainingDataWriter<GameState_, Tensorizor_>::GameData_sptr
TrainingDataWriter<GameState_, Tensorizor_>::get_data(game_id_t id) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = game_data_map_.find(id);
  if (it == game_data_map_.end()) {
    GameData_sptr ptr(new GameData(id));
    game_data_map_[id] = ptr;
    return ptr;
  }
  return it->second;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::close(GameData_sptr data) {
  if (data->closed()) return;
  data->close();

  std::unique_lock<std::mutex> lock(mutex_);
  game_queue_[queue_index_].push_back(data);
  game_data_map_.erase(data->id());
  lock.unlock();
  cv_.notify_one();
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::shut_down() {
  closed_ = true;
  cv_.notify_one();
  if (thread_->joinable()) thread_->join();
  delete thread_;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>::TrainingDataWriter(const Params& params)
: params_(params)
{
  namespace bf = boost::filesystem;

  if (params.clear_dir && bf::is_directory(games_dir())) {
    bf::remove_all(games_dir());
  }
  bf::create_directories(games_dir());

  thread_ = new std::thread([&] { loop(); });
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>::~TrainingDataWriter() {
  shut_down();
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::loop() {
  while (!closed_) {
    std::unique_lock<std::mutex> lock(mutex_);
    game_queue_t& queue = game_queue_[queue_index_];
    cv_.wait(lock, [&]{ return !queue.empty() || closed_;});
    queue_index_ = 1 - queue_index_;
    lock.unlock();
    for (GameData_sptr& data : queue) {
      write_to_file(data.get());
    }
    queue.clear();
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::write_to_file(const GameData* data) {
  int rows = 0;
  for (const DataChunk& chunk : data->chunks()) {
    rows += chunk.rows();
  }

  InputBlob input(input_shape(rows));
  PolicyBlob policy(policy_shape(rows));
  ValueBlob value(value_shape(rows));

  using namespace torch::indexing;
  for (const DataChunk& chunk : data->chunks()) {
    auto write_slice = Slice(0, rows);
    auto read_slice = Slice(0, chunk.rows());

    input.asTorch().index_put_({write_slice}, chunk.input().asTorch().index({read_slice}));
    policy.asTorch().index_put_({write_slice}, chunk.policy().asTorch().index({read_slice}));
    value.asTorch().index_put_({write_slice}, chunk.value().asTorch().index({read_slice}));
  }

  int64_t ns_since_epoch = util::ns_since_epoch(std::chrono::steady_clock::now());
  std::string output_filename = util::create_string("%ld-%d.ptd", ns_since_epoch, rows);
  std::string tmp_output_filename = util::create_string(".%s", output_filename.c_str());
  boost::filesystem::path output_path = games_dir() / output_filename;
  boost::filesystem::path tmp_output_path = games_dir() / tmp_output_filename;

  auto slice = torch::indexing::Slice(torch::indexing::None, rows);
  using tensor_map_t = std::map<std::string, torch::Tensor>;

  tensor_map_t tensor_map;
  tensor_map["input"] = input.asTorch().index({slice});
  tensor_map["policy"] = policy.asTorch().index({slice});
  tensor_map["value"] = value.asTorch().index({slice});

  // write-then-mv to avoid race-conditions with partially-written files
  torch_util::save(tensor_map, tmp_output_path.string());
  std::filesystem::rename(tmp_output_path.c_str(), output_path.c_str());
}

}  // namespace common
