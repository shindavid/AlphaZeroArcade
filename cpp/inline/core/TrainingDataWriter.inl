#include <core/TrainingDataWriter.hpp>

#include <filesystem>
#include <map>
#include <string>

#include <core/GameServer.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>
#include <util/StringUtil.hpp>
#include <util/TorchUtil.hpp>

namespace core {

template <bool ApplySymmetry>
struct AuxTargetHelper {};

template <>
struct AuxTargetHelper<false> {
  template <eigen_util::FixedTensorConcept Tensor, core::GameStateConcept GameState>
  static void apply(Tensor&, const GameState&, symmetry_index_t) {}
};

template <>
struct AuxTargetHelper<true> {
  template <eigen_util::FixedTensorConcept Tensor, core::GameStateConcept GameState>
  static void apply(Tensor& tensor, const GameState& state, symmetry_index_t sym_index) {
    state.template get_symmetry<Tensor>(sym_index)->apply(tensor);
  }
};

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>*
    TrainingDataWriter<GameState_, Tensorizor_>::instance_ = nullptr;

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
auto TrainingDataWriter<GameState_, Tensorizor_>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("TrainingDataWriter options");
  return desc
      .template add_option<"max-rows", 'M'>(
          po::value<int64_t>(&max_rows)->default_value(max_rows),
          "if specified, kill process after writing this many rows");
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::DataChunk() {
  tensors_ = new TensorGroup[kRowsPerChunk];
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::~DataChunk() {
  delete[] tensors_;
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
typename TrainingDataWriter<GameState_, Tensorizor_>::TensorGroup&
TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::get_next_group() {
  return tensors_[rows_++];
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::record_for_all(
    const GameState& state, const GameOutcome& value) {
  for (int i = 0; i < rows_; ++i) {
    TensorGroup& group = tensors_[i];
    GameOutcome shifted_value = value;
    eigen_util::left_rotate(shifted_value, group.current_player);
    group.value = eigen_util::reinterpret_as_tensor<ValueTensor>(shifted_value);

    mp::constexpr_for<0, kNumAuxTargets, 1>([&](auto i) {
      using AuxTarget = mp::TypeAt_t<AuxTargetList, i>;
      auto& aux_tensor = std::get<i>(group.aux_targets);
      AuxTarget::tensorize(aux_tensor, state, group.state.get_current_player());
      AuxTargetHelper<AuxTarget::kApplySymmetry>::apply(aux_tensor, group.state, group.sym_index);
    });
  }
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
typename TrainingDataWriter<GameState_, Tensorizor_>::TensorGroup&
TrainingDataWriter<GameState_, Tensorizor_>::GameData::get_next_group() {
  std::unique_lock<std::mutex> lock(mutex_);
  return get_next_chunk()->get_next_group();
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
typename TrainingDataWriter<GameState_, Tensorizor_>::DataChunk*
TrainingDataWriter<GameState_, Tensorizor_>::GameData::get_next_chunk() {
  if (chunks_.empty() || chunks_.back().full()) {
    chunks_.emplace_back();
  }
  return &chunks_.back();
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::GameData::record_for_all(
    const GameState& state, const GameOutcome& value) {
  std::unique_lock<std::mutex> lock(mutex_);
  for (DataChunk& chunk : chunks_) {
    chunk.record_for_all(state, value);
  }
  pending_groups_.clear();
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::GameData::commit_opp_reply_to_pending_groups(
    const PolicyTensor& opp_policy) {
  for (auto& group : pending_groups_) {
    const GameState& state = group->state;
    symmetry_index_t sym_index = group->sym_index;
    auto* policy_transform = state.template get_symmetry<PolicyTensor>(sym_index);

    group->opp_policy = opp_policy;
    policy_transform->apply(group->opp_policy);
  }
  pending_groups_.clear();
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
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

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
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

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::close(GameData_sptr data) {
  if (data->closed()) return;
  data->close();

  std::unique_lock<std::mutex> lock(mutex_);
  completed_games_[queue_index_].push_back(data);
  game_data_map_.erase(data->id());
  lock.unlock();
  cv_.notify_one();
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::shut_down() {
  closed_ = true;
  cv_.notify_one();
  if (thread_->joinable()) thread_->join();
  delete thread_;
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::pause() {
  LOG_INFO << "TrainingDataWriter: pausing";
  std::unique_lock lock(mutex_);
  if (paused_) {
    LOG_INFO << "TrainingDataWriter: handle_pause_receipt (already paused)";
    core::LoopControllerClient::get()->handle_pause_receipt();
    return;
  }
  paused_ = true;
  lock.unlock();
  cv_.notify_one();
  LOG_INFO << "TrainingDataWriter: pause complete!";
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::unpause() {
  LOG_INFO << "TrainingDataWriter: unpausing";
  std::unique_lock lock(mutex_);
  if (!paused_) {
    LOG_INFO << "TrainingDataWriter: handle_unpause_receipt (already unpaused)";
    core::LoopControllerClient::get()->handle_unpause_receipt();
    return;
  }
  paused_ = false;
  lock.unlock();
  cv_.notify_one();
  LOG_INFO << "TrainingDataWriter: unpause complete!";
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>::TrainingDataWriter(const Params& params)
    : params_(params) {
  if (LoopControllerClient::initialized()) {
    LoopControllerClient::get()->add_listener(this);
  }
  thread_ = new std::thread([&] { loop(); });
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>::~TrainingDataWriter() {
  shut_down();
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::loop() {
  while (!closed_) {
    std::unique_lock lock(mutex_);
    game_queue_t& queue = completed_games_[queue_index_];
    cv_.wait(lock, [&] { return !queue.empty() || closed_ || paused_; });
    if (paused_) {
      LOG_INFO << "TrainingDataWriter: handle_pause_receipt";
      core::LoopControllerClient::get()->handle_pause_receipt();
      cv_.wait(lock, [&] { return !paused_; });
      LOG_INFO << "TrainingDataWriter: handle_unpause_receipt";
      core::LoopControllerClient::get()->handle_unpause_receipt();
    }
    queue_index_ = 1 - queue_index_;
    lock.unlock();
    for (GameData_sptr& data : queue) {
      if (send(data.get())) break;
    }
    queue.clear();
  }
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
bool TrainingDataWriter<GameState_, Tensorizor_>::send(const GameData* data) {
  int total_rows = 0;
  for (const DataChunk& chunk : data->chunks()) {
    total_rows += chunk.rows();
  }

  auto input_shape =
      util::to_std_array<int64_t>(total_rows, eigen_util::to_int64_std_array_v<InputShape>);
  auto policy_shape =
      util::to_std_array<int64_t>(total_rows, eigen_util::to_int64_std_array_v<PolicyShape>);
  auto value_shape =
      util::to_std_array<int64_t>(total_rows, eigen_util::to_int64_std_array_v<ValueShape>);

  torch::Tensor input = torch::empty(input_shape, torch_util::to_dtype_v<InputScalar>);
  torch::Tensor policy = torch::empty(policy_shape, torch_util::to_dtype_v<PolicyScalar>);
  torch::Tensor opp_policy = torch::empty(policy_shape, torch_util::to_dtype_v<PolicyScalar>);
  torch::Tensor value = torch::empty(value_shape, torch_util::to_dtype_v<ValueScalar>);

  AuxTargetTorchTensorTuple aux_targets;
  mp::constexpr_for<0, kNumAuxTargets, 1>([&](auto i) {
    using AuxTarget = mp::TypeAt_t<AuxTargetList, i>;
    using AuxTensor = typename AuxTarget::Tensor;
    using AuxShape = typename AuxTarget::Shape;
    using AuxScalar = torch_util::convert_type_t<typename AuxTensor::Scalar>;
    auto& aux_tgt = std::get<i>(aux_targets);
    auto aux_shape =
        util::to_std_array<int64_t>(total_rows, eigen_util::to_int64_std_array_v<AuxShape>);
    aux_tgt = torch::empty(aux_shape, torch_util::to_dtype_v<AuxScalar>);
  });

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
      memcpy(policy_data + policy_size * rows, group.policy.data(),
             policy_size * sizeof(PolicyScalar));
      memcpy(opp_policy_data + policy_size * rows, group.opp_policy.data(),
             policy_size * sizeof(PolicyScalar));
      memcpy(value_data + value_size * rows, group.value.data(), value_size * sizeof(ValueScalar));

      mp::constexpr_for<0, kNumAuxTargets, 1>([&](auto i) {
        using AuxTarget = mp::TypeAt_t<AuxTargetList, i>;
        using AuxTensor = typename AuxTarget::Tensor;
        using AuxShape = typename AuxTarget::Shape;
        using AuxScalar = torch_util::convert_type_t<typename AuxTensor::Scalar>;
        constexpr size_t aux_size = AuxShape::total_size;

        const auto& aux_src = std::get<i>(group.aux_targets);
        auto& aux_tgt = std::get<i>(aux_targets);
        AuxScalar* aux_data = aux_tgt.template data_ptr<AuxScalar>();

        memcpy(aux_data + aux_size * rows, aux_src.data(), aux_size * sizeof(AuxScalar));
      });

      rows++;
    }
  }

  int64_t start_timestamp = data->start_timestamp();
  int64_t cur_timestamp = util::ns_since_epoch();

  core::LoopControllerClient* client = core::LoopControllerClient::get();
  util::release_assert(client, "TrainingDataWriter: no LoopControllerClient");

  int model_generation = client ? client->cur_generation() : 0;

  using tensor_map_t = std::map<std::string, torch::Tensor>;

  tensor_map_t tensor_map;
  tensor_map["input"] = input;
  tensor_map["policy"] = policy;
  tensor_map["opp_policy"] = opp_policy;
  tensor_map["value"] = value;

  mp::constexpr_for<0, kNumAuxTargets, 1>([&](auto i) {
    using AuxTarget = mp::TypeAt_t<AuxTargetList, i>;
    tensor_map[AuxTarget::kName] = std::get<i>(aux_targets);
  });

  std::stringstream ss;
  torch_util::save(tensor_map, ss);

  auto new_rows_written = rows_written_ + rows;
  bool done = params_.max_rows > 0 && new_rows_written >= params_.max_rows;
  bool flush = done || client->ready_for_games_flush(cur_timestamp);

  boost::json::object msg;
  msg["type"] = "game";
  msg["gen"] = model_generation;
  msg["start_timestamp"] = start_timestamp;
  msg["end_timestamp"] = cur_timestamp;
  msg["rows"] = rows;
  msg["flush"] = flush;

  if (flush) {
    if (client->report_metrics()) {
      msg["metrics"] = client->get_perf_stats().to_json();
    }
    client->set_last_games_flush_ts(cur_timestamp);
  }
  client->send_with_file(msg, ss);

  rows_written_ = new_rows_written;
  if (done) {
    std::cout << "TrainingDataWriter: shutting down after writing " << rows_written_ << " rows"
              << std::endl;
    closed_ = true;

    // This assumes that we are in the same process as the GameServer, which is true for now. I
    // don't foresee the assumption being violated.
    GameServer<GameState>::request_shutdown();
    return true;
  }
  return false;
}

}  // namespace core
