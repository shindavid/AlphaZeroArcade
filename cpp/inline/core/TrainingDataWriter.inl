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

namespace detail {

inline boost::filesystem::path make_games_sub_dir(int model_generation) {
  return util::create_string("gen-%d", model_generation);
}

}  // namespace detail

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
      .template add_option<"games-base-dir", 'D'>(
          po::value<std::string>(&games_base_dir),
          "base directory for games files. This is fixed. Game files look like "
          "{games-base-dir}/gen-{generation}/{timestamp}.ptj")
      .template add_option<"max-rows", 'M'>(
          po::value<int64_t>(&max_rows)->default_value(max_rows),
          "if specified, kill process after writing this many rows")
      .template add_flag<"report-metrics", "do-not-report-metrics">(
          &report_metrics, "report metrics to cmd-server periodically",
          "do not report metrics to cmd-server");
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
  std::unique_lock lock(mutex_);
  paused_ = true;
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::unpause() {
  std::unique_lock lock(mutex_);
  paused_ = false;
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>::TrainingDataWriter(const Params& params)
    : params_(params), games_base_dir_(params.games_base_dir) {
  util::clean_assert(params.games_base_dir.size() > 0,
                     "TrainingDataWriter: games_base_dir must be specified");

  if (CmdServerClient::initialized()) {
    CmdServerClient::get()->add_listener(this);
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
      core::CmdServerClient::get()->notify_pause_received(this);
      cv_.wait(lock, [&] { return !paused_; });
    }
    queue_index_ = 1 - queue_index_;
    lock.unlock();
    for (GameData_sptr& data : queue) {
      write_to_file(data.get());
    }
    queue.clear();
  }
}

template <GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::write_to_file(const GameData* data) {
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
  std::string output_filename = util::create_string("%ld.ptd", cur_timestamp);
  std::string tmp_output_filename = util::create_string(".%s", output_filename.c_str());

  core::CmdServerClient* client = core::CmdServerClient::get();

  int model_generation = client ? client->cur_generation() : 0;

  boost::filesystem::path games_sub_dir = detail::make_games_sub_dir(model_generation);
  boost::filesystem::path full_games_dir = games_base_dir_ / games_sub_dir;
  boost::filesystem::path output_path = full_games_dir / output_filename;
  boost::filesystem::path tmp_output_path = full_games_dir / tmp_output_filename;

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

  // write-then-mv to avoid race-conditions with partially-written files
  torch_util::save(tensor_map, tmp_output_path.string());
  std::filesystem::rename(tmp_output_path.c_str(), output_path.c_str());

  // inform cmd-server of new file
  if (client) {
    boost::json::object msg;
    msg["type"] = "game";
    msg["gen"] = model_generation;
    msg["start_timestamp"] = start_timestamp;
    msg["end_timestamp"] = cur_timestamp;
    msg["rows"] = rows;

    if (params_.report_metrics && client->ready_for_metrics(cur_timestamp)) {
      msg["metrics"] = client->get_perf_stats().to_json();
      client->set_last_metrics_ts(cur_timestamp);
    }
    client->send(msg);
  }

  rows_written_ += rows;
  if (params_.max_rows > 0 && rows_written_ >= params_.max_rows) {
    std::cout << "TrainingDataWriter: wrote " << rows_written_ << " rows, exiting" << std::endl;

    if (client) {
      boost::json::object msg;
      msg["type"] = "max_rows_reached";
      client->send(msg);
    }

    // This assumes that we are in the same process as the GameServer, which is true for now. I
    // don't foresee the assumption being violated.
    GameServer<GameState>::request_shutdown();
  }
}

}  // namespace core
