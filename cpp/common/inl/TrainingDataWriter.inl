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
TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::DataChunk() {
  tensors_ = new TensorGroup[kRowsPerChunk];
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::~DataChunk() {
  delete[] tensors_;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
typename TrainingDataWriter<GameState_, Tensorizor_>::TensorGroup&
TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::get_next_group() {
  return tensors_[rows_++];
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::record_for_all(const GameOutcome& value) {
  for (int i = 0; i < rows_; ++i) {
    TensorGroup& group = tensors_[i];
    GameOutcome shifted_value = value;
    eigen_util::left_rotate(shifted_value, group.current_player);
    group.value = eigen_util::reinterpret_as_tensor<ValueTensor>(shifted_value);
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
typename TrainingDataWriter<GameState_, Tensorizor_>::TensorGroup&
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
  pending_groups_.clear();
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::GameData::add_pending_group(
    SymmetryTransform* transform, TensorGroup* group) {
  pending_groups_.emplace_back(transform, group);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::GameData::commit_opp_reply_to_pending_groups(
    const PolicyTensor& opp_policy)
{
  for (auto& transform_group : pending_groups_) {
    auto* transform = transform_group.transform;
    auto* group = transform_group.group;

    group->opp_policy = opp_policy;
    transform->transform_policy(group->opp_policy);
  }
  pending_groups_.clear();
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
  int total_rows = 0;
  for (const DataChunk& chunk : data->chunks()) {
    total_rows += chunk.rows();
  }

  auto input_shape = util::to_std_array<int64_t>(total_rows, eigen_util::to_int64_std_array_v<InputShape>);
  auto policy_shape = util::to_std_array<int64_t>(total_rows, eigen_util::to_int64_std_array_v<PolicyShape>);
  auto value_shape = util::to_std_array<int64_t>(total_rows, eigen_util::to_int64_std_array_v<ValueShape>);

  torch::Tensor input = torch::empty(input_shape, torch_util::to_dtype_v<InputScalar>);
  torch::Tensor policy = torch::empty(policy_shape, torch_util::to_dtype_v<PolicyScalar>);
  torch::Tensor opp_policy = torch::empty(policy_shape, torch_util::to_dtype_v<PolicyScalar>);
  torch::Tensor value = torch::empty(value_shape, torch_util::to_dtype_v<ValueScalar>);

  constexpr size_t input_size = InputShape::total_size;
  constexpr size_t policy_size = PolicyShape::total_size;
  constexpr size_t value_size = ValueShape::total_size;

  InputScalar* input_data = input.data_ptr<InputScalar>();
  PolicyScalar* policy_data = policy.data_ptr<PolicyScalar>();
  PolicyScalar* opp_policy_data = opp_policy.data_ptr<PolicyScalar>();
  ValueScalar* value_data = value.data_ptr<ValueScalar>();

  int rows = 0;
  for (const DataChunk& chunk : data->chunks()) {
    for (int r = 0; r < chunk.rows(); ++r) {
      const TensorGroup& group = chunk.get_group(r);

      memcpy(input_data + input_size * rows, group.input.data(), input_size * sizeof(InputScalar));
      memcpy(policy_data + policy_size * rows, group.policy.data(), policy_size * sizeof(PolicyScalar));
      memcpy(opp_policy_data + policy_size * rows, group.opp_policy.data(), policy_size * sizeof(PolicyScalar));
      memcpy(value_data + value_size * rows, group.value.data(), value_size * sizeof(ValueScalar));

      rows++;
    }
  }

  int64_t ns_since_epoch = util::ns_since_epoch(std::chrono::steady_clock::now());
  std::string output_filename = util::create_string("%ld-%d.ptd", ns_since_epoch, rows);
  std::string tmp_output_filename = util::create_string(".%s", output_filename.c_str());
  boost::filesystem::path output_path = games_dir() / output_filename;
  boost::filesystem::path tmp_output_path = games_dir() / tmp_output_filename;

  using tensor_map_t = std::map<std::string, torch::Tensor>;

  tensor_map_t tensor_map;
  tensor_map["input"] = input;
  tensor_map["policy"] = policy;
  tensor_map["opp_policy"] = opp_policy;
  tensor_map["value"] = value;

  // write-then-mv to avoid race-conditions with partially-written files
  torch_util::save(tensor_map, tmp_output_path.string());
  std::filesystem::rename(tmp_output_path.c_str(), output_path.c_str());
}

}  // namespace common
