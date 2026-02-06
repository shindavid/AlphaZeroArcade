#include "search/GameLog.hpp"

#include "util/Asserts.hpp"
#include "util/IndexedDispatcher.hpp"
#include "util/MetaProgramming.hpp"

#include <algorithm>

namespace search {

template <search::concepts::Traits Traits>
GameReadLog<Traits>::DataLayout::DataLayout(const GameLogMetadata& m) {
  final_state = 0;
  outcome = align(final_state + sizeof(State));
  sampled_indices_start = align(outcome + sizeof(GameResultTensor));
  mem_offsets_start = align(sampled_indices_start + sizeof(pos_index_t) * m.num_samples);
  records_start = align(mem_offsets_start + sizeof(mem_offset_t) * m.num_positions);
}

template <search::concepts::Traits Traits>
GameReadLog<Traits>::GameReadLog(const char* filename, int game_index,
                                 const GameLogMetadata& metadata, const char* buffer)
    : filename_(filename),
      game_index_(game_index),
      metadata_(metadata),
      buffer_(buffer),
      layout_(metadata) {
  RELEASE_ASSERT(num_positions() > 0, "Empty game log file {}[{}]", filename, game_index);
}

template <search::concepts::Traits Traits>
ShapeInfo* GameReadLog<Traits>::get_input_shapes() {
  constexpr int n_inputs = 1;
  constexpr int n = n_inputs + 1;  // +1 for terminator
  using InputShape = InputTensor::Dimensions;

  ShapeInfo* info_array = new ShapeInfo[n];
  info_array[0].template init<InputShape>("input", 0);

  return info_array;
}

template <search::concepts::Traits Traits>
ShapeInfo* GameReadLog<Traits>::get_target_shapes() {
  constexpr int n_targets = mp::Length_v<TrainingTargets>;
  constexpr int n = n_targets + 1;  // +1 for terminator

  ShapeInfo* info_array = new ShapeInfo[n];

  mp::constexpr_for<0, n_targets, 1>([&](auto a) {
    using Target = mp::TypeAt_t<TrainingTargets, a>;
    using Shape = Target::Tensor::Dimensions;
    info_array[a].template init<Shape>(Target::kName, a);
  });

  return info_array;
}

template <search::concepts::Traits Traits>
ShapeInfo* GameReadLog<Traits>::get_head_shapes() {
  constexpr int n_heads = mp::Length_v<NetworkHeads>;
  constexpr int n = n_heads + 1;  // +1 for terminator

  ShapeInfo* info_array = new ShapeInfo[n];

  mp::constexpr_for<0, n_heads, 1>([&](auto a) {
    using Head = mp::TypeAt_t<NetworkHeads, a>;
    using Shape = Head::Tensor::Dimensions;
    info_array[a].template init<Shape>(Head::kName, a);
  });

  return info_array;
}

template <search::concepts::Traits Traits>
void GameReadLog<Traits>::load(int row_index, bool apply_symmetry,
                               const std::vector<int>& target_indices, float* output_array) const {
  RELEASE_ASSERT(row_index >= 0 && row_index < num_sampled_positions(),
                 "Index {} out of bounds [0, {}) in {}[{}]", row_index, num_sampled_positions(),
                 filename_, game_index_);

  pos_index_t state_index = get_pos_index(row_index);
  const GameLogCompactRecord* record = &get_record(get_mem_offset(state_index));

  const GameLogCompactRecord* next_record = nullptr;
  if (state_index + 1 < num_positions()) {
    next_record = &get_record(get_mem_offset(state_index + 1));
  }

  int num_prev_states_to_cp = std::min(InputTensorizor::kNumStatesToEncode - 1, state_index);
  int num_states = num_prev_states_to_cp + 1;

  State states[num_states];
  for (int i = 0; i < num_prev_states_to_cp; ++i) {
    int prev_state_index = state_index - num_prev_states_to_cp + i;
    states[i] = get_record(get_mem_offset(prev_state_index)).position;
  }
  states[num_states - 1] = record->position;

  State* cur_pos = &states[num_states - 1];
  State final_state = get_final_state();

  group::element_t sym = 0;
  if (apply_symmetry) {
    sym = Game::Symmetries::get_mask(*cur_pos).choose_random_on_index();
  }

  for (int i = 0; i < num_states; ++i) {
    Game::Symmetries::apply(states[i], sym);
  }
  Game::Symmetries::apply(final_state, sym);

  GameLogViewParams params;
  params.record = record;
  params.next_record = next_record;
  params.cur_pos = cur_pos;
  params.final_pos = &final_state;
  params.outcome = &get_outcome();
  params.sym = sym;

  GameLogView view;
  Algorithms::to_view(params, view);

  InputTensorizor input_tensorizor;
  for (int i = 0; i < num_states; ++i) {
    input_tensorizor.update(states[i]);
  }
  auto input = input_tensorizor.tensorize();

  constexpr int kInputSize = InputTensorizor::Tensor::Dimensions::total_size;
  output_array = std::copy(input.data(), input.data() + kInputSize, output_array);

  constexpr size_t N = mp::Length_v<TrainingTargets>;
  for (int target_index : target_indices) {
    util::IndexedDispatcher<N>::call(target_index, [&](auto t) {
      using Target = mp::TypeAt_t<TrainingTargets, t>;
      using Tensor = Target::Tensor;
      constexpr int kSize = Tensor::Dimensions::total_size;

      Tensor tensor;
      bool mask = Target::tensorize(view, tensor);
      output_array = std::copy(tensor.data(), tensor.data() + kSize, output_array);
      output_array[0] = mask;
      output_array++;
    });
  }
}

template <search::concepts::Traits Traits>
const typename GameReadLog<Traits>::State& GameReadLog<Traits>::get_final_state() const {
  return *reinterpret_cast<const State*>(buffer_ + layout_.final_state);
}

template <search::concepts::Traits Traits>
const typename GameReadLog<Traits>::GameResultTensor& GameReadLog<Traits>::get_outcome() const {
  return *reinterpret_cast<const GameResultTensor*>(buffer_ + layout_.outcome);
}

template <search::concepts::Traits Traits>
GameLogCommon::pos_index_t GameReadLog<Traits>::get_pos_index(int index) const {
  const pos_index_t* ptr = (const pos_index_t*)&buffer_[layout_.sampled_indices_start];
  return ptr[index];
}

template <search::concepts::Traits Traits>
const typename GameReadLog<Traits>::GameLogCompactRecord& GameReadLog<Traits>::get_record(
  mem_offset_t mem_offset) const {
  const GameLogCompactRecord* ptr =
    (const GameLogCompactRecord*)&buffer_[layout_.records_start + mem_offset];
  return *ptr;
}

template <search::concepts::Traits Traits>
typename GameReadLog<Traits>::mem_offset_t GameReadLog<Traits>::get_mem_offset(
  int state_index) const {
  const mem_offset_t* mem_offsets_ptr = (const mem_offset_t*)&buffer_[layout_.mem_offsets_start];
  return mem_offsets_ptr[state_index];
}

template <search::concepts::Traits Traits>
GameWriteLog<Traits>::GameWriteLog(core::game_id_t id, int64_t start_timestamp)
    : id_(id), start_timestamp_(start_timestamp) {}

template <search::concepts::Traits Traits>
GameWriteLog<Traits>::~GameWriteLog() {
  for (GameLogFullRecord* full_record : full_records_) {
    delete full_record;
  }
}

template <search::concepts::Traits Traits>
void GameWriteLog<Traits>::add(const TrainingInfo& training_info) {
  bool use_for_training = training_info.use_for_training;

  // TODO: get GameLogFullRecord objects from an object pool
  GameLogFullRecord* full_record = new GameLogFullRecord();
  Algorithms::to_record(training_info, *full_record);
  full_records_.push_back(full_record);
  sample_count_ += use_for_training;
}

template <search::concepts::Traits Traits>
void GameWriteLog<Traits>::add_terminal(const State& state, const GameResultTensor& outcome) {
  terminal_added_ = true;
  final_state_ = state;
  outcome_ = outcome;
}

template <search::concepts::Traits Traits>
bool GameWriteLog<Traits>::was_previous_entry_used_for_policy_training() const {
  if (full_records_.empty()) {
    return false;
  }
  return full_records_.back()->use_for_training;
}

template <search::concepts::Traits Traits>
GameLogMetadata GameLogSerializer<Traits>::serialize(const GameWriteLog* log,
                                                     std::vector<char>& buf, int client_id) {
  uint32_t start_buf_size = buf.size();
  RELEASE_ASSERT(log->terminal_added_);
  int num_full_records = log->full_records_.size();

  for (int move_num = 0; move_num < num_full_records; ++move_num) {
    const GameLogFullRecord* full_record = log->full_records_[move_num];

    mem_offsets_.push_back(data_buf_.size());
    if (full_record->use_for_training) {
      sampled_indices_.push_back(move_num);
    }

    Algorithms::serialize_record(*full_record, data_buf_);
  }

  GameLogCommon::write_section(buf, &log->final_state_);
  GameLogCommon::write_section(buf, &log->outcome_);
  GameLogCommon::write_section(buf, sampled_indices_.data(), sampled_indices_.size());
  GameLogCommon::write_section(buf, mem_offsets_.data(), mem_offsets_.size());
  GameLogCommon::write_section(buf, data_buf_.data(), data_buf_.size());

  // clear vectors
  sampled_indices_.clear();
  mem_offsets_.clear();
  data_buf_.clear();

  uint32_t end_buf_size = buf.size();

  // NOTE: the start_offset value is initially assigned here relative to the start of the GameData
  // region. It will be updated later to be relative to the start of the file.
  GameLogMetadata metadata;
  metadata.start_timestamp = log->start_timestamp_;
  metadata.start_offset = start_buf_size;
  metadata.data_size = end_buf_size - start_buf_size;
  metadata.num_samples = log->sample_count_;
  metadata.num_positions = num_full_records;
  metadata.client_id = client_id;

  return metadata;
}

}  // namespace search
