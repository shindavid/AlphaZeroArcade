#include <common/TrainingDataWriter.hpp>

#include <map>
#include <string>

#include <util/StringUtil.hpp>
#include <util/TorchUtil.hpp>

namespace common {

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::DataChunk()
: input_(input_shape(kRowsPerChunk))
, policy_(policy_shape(kRowsPerChunk))
, value_(value_shape(kRowsPerChunk)) {}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
typename TrainingDataWriter<GameState_, Tensorizor_>::EigenSlab
TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::get_next_slab() {
  int rows = rows_++;
  return EigenSlab{
    input_.template eigenSlab<typename TensorizorTypes::Shape<1>>(rows),
    policy_.eigenSlab(rows),
    value_.eigenSlab(rows)
    };
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::DataChunk::record_for_all(const ValueEigenSlab& value) {
  for (int i = 0; i < rows_; ++i) {
    value_.eigenSlab(i) = value;
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
auto TrainingDataWriter<GameState_, Tensorizor_>::TrainingDataWriter::input_shape(int rows) {
  return util::to_std_array<int>(rows, util::std_array_v<int, TensorShape>);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
auto TrainingDataWriter<GameState_, Tensorizor_>::TrainingDataWriter::policy_shape(int rows) {
  return util::to_std_array<int>(rows, GameState::kNumGlobalActions);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
auto TrainingDataWriter<GameState_, Tensorizor_>::TrainingDataWriter::value_shape(int rows) {
  return util::to_std_array<int>(rows, GameState::kNumPlayers);
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
typename TrainingDataWriter<GameState_, Tensorizor_>::EigenSlab
TrainingDataWriter<GameState_, Tensorizor_>::GameData::get_next_slab() {
  return get_next_chunk()->get_next_slab();
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
void TrainingDataWriter<GameState_, Tensorizor_>::GameData::record_for_all(const ValueEigenSlab& value) {
  for (DataChunk& chunk : chunks_) {
    chunk.record_for_all(value);
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>::TrainingDataWriter(const boost::filesystem::path& output_path)
: output_path_(output_path)
, input_(input_shape(kRowsPerFile))
, policy_(kRowsPerFile, PolicySlab::Cols, policy_shape(kRowsPerFile))
, value_(kRowsPerFile, ValueSlab::Cols, value_shape(kRowsPerFile))
{
  namespace bf = boost::filesystem;

  if (bf::is_directory(output_path)) {
    bf::remove_all(output_path);
  }
  bf::create_directories(output_path);

  thread_ = new std::thread([&] { loop(); });
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
TrainingDataWriter<GameState_, Tensorizor_>::~TrainingDataWriter() {
  shut_down();
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::close(const GameData* data) {
  std::unique_lock<std::mutex> lock(mutex_);
  game_queue_[queue_index_].push_back(data);
  lock.unlock();
  cv_.notify_one();
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::shut_down() {
  closed_ = true;
  cv_.notify_one();
  if (thread_->joinable()) thread_->join();
  delete thread_;
  flush();
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::flush() {
  if (rows_ == 0) return;

  std::string output_filename = util::create_string("%d.pt", file_number_);
  boost::filesystem::path output_path = output_path_ / output_filename;

  auto slice = torch::indexing::Slice(torch::indexing::None, rows_);
  using tensor_map_t = std::map<std::string, torch::Tensor>;

  tensor_map_t tensor_map;
  tensor_map["input"] = input_.asTorch().index({slice});
  tensor_map["policy"] = policy_.asTorch().index({slice});
  tensor_map["value"] = value_.asTorch().index({slice});
  torch_util::save(tensor_map, output_path.string());

  file_number_++;
  rows_ = 0;
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::loop() {
  while (!closed_) {
    std::unique_lock<std::mutex> lock(mutex_);
    game_queue_t& queue = game_queue_[queue_index_];
    cv_.wait(lock, [&]{ return !queue.empty() || closed_;});
    queue_index_ = 1 - queue_index_;
    lock.unlock();
    for (const GameData* data : queue) {
      for (const DataChunk& chunk : data->chunks()) {
        write(chunk);
      }
      delete data;
    }
    queue.clear();
  }
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::write(const DataChunk& chunk) {
  int rows_to_write = std::min(chunk.rows(), kRowsPerFile - rows_);
  partial_write(chunk, 0, rows_to_write);
  partial_write(chunk, rows_to_write, chunk.rows());
}

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
void TrainingDataWriter<GameState_, Tensorizor_>::partial_write(const DataChunk& chunk, int start, int end) {
  int num_rows_to_write = end - start;
  if (num_rows_to_write <= 0) return;

  using namespace torch::indexing;
  auto write_slice = Slice(rows_, rows_ + num_rows_to_write);
  auto read_slice = Slice(start, end);

  int new_rows = rows_ + num_rows_to_write;
  input_.asTorch().index_put_({write_slice}, chunk.input().asTorch().index({read_slice}));
  policy_.asTorch().index_put_({write_slice}, chunk.policy().asTorch().index({read_slice}));
  value_.asTorch().index_put_({write_slice}, chunk.value().asTorch().index({read_slice}));

  rows_ = new_rows;
  if (rows_ == kRowsPerFile) {
    flush();
  }
}

}  // namespace common
