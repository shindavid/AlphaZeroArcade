#include "beta0/GameLog.hpp"

#include "util/Asserts.hpp"
#include "util/IndexedDispatcher.hpp"
#include "util/MetaProgramming.hpp"

#include <algorithm>

namespace beta0 {

template <::beta0::concepts::Spec Spec>
GameReadLog<Spec>::DataLayout::DataLayout(const search::GameLogMetadata& m) {
  final_frame = 0;
  outcome = align(final_frame + sizeof(InputFrame));
  sampled_indices_start = align(outcome + sizeof(GameResultTensor));
  mem_offsets_start = align(sampled_indices_start + sizeof(frame_index_t) * m.num_samples);
  records_start = align(mem_offsets_start + sizeof(mem_offset_t) * m.num_frames);
}

template <::beta0::concepts::Spec Spec>
GameReadLog<Spec>::GameReadLog(const char* filename, int game_index,
                               const search::GameLogMetadata& metadata, const char* buffer)
    : filename_(filename),
      game_index_(game_index),
      metadata_(metadata),
      buffer_(buffer),
      layout_(metadata) {
  RELEASE_ASSERT(num_frames() > 0, "Empty game log file {}[{}]", filename, game_index);
}

template <::beta0::concepts::Spec Spec>
search::ShapeInfo* GameReadLog<Spec>::get_input_shapes() {
  constexpr int n_inputs = 1;
  constexpr int n = n_inputs + 1;  // +1 for terminator
  using InputShape = InputTensor::Dimensions;

  search::ShapeInfo* info_array = new search::ShapeInfo[n];
  info_array[0].template init<InputShape>("input", 0);

  return info_array;
}

template <::beta0::concepts::Spec Spec>
search::ShapeInfo* GameReadLog<Spec>::get_target_shapes() {
  constexpr int n_targets = mp::Length_v<TrainingTargets>;
  constexpr int n = n_targets + 1;  // +1 for terminator

  search::ShapeInfo* info_array = new search::ShapeInfo[n];

  mp::constexpr_for<0, n_targets, 1>([&](auto a) {
    using Target = mp::TypeAt_t<TrainingTargets, a>;
    using Shape = Target::Tensor::Dimensions;
    info_array[a].template init<Shape>(Target::kName, a);
  });

  return info_array;
}

template <::beta0::concepts::Spec Spec>
search::ShapeInfo* GameReadLog<Spec>::get_head_shapes() {
  constexpr int n_heads = mp::Length_v<NetworkHeads>;
  constexpr int n = n_heads + 1;  // +1 for terminator

  search::ShapeInfo* info_array = new search::ShapeInfo[n];

  mp::constexpr_for<0, n_heads, 1>([&](auto a) {
    using Head = mp::TypeAt_t<NetworkHeads, a>;
    using Shape = Head::Tensor::Dimensions;
    info_array[a].template init<Shape>(Head::kName, a);
  });

  return info_array;
}

template <::beta0::concepts::Spec Spec>
void GameReadLog<Spec>::load(int row_index, bool apply_symmetry,
                             const std::vector<int>& target_indices, float* output_array) const {
  RELEASE_ASSERT(row_index >= 0 && row_index < num_sampled_frames(),
                 "Index {} out of bounds [0, {}) in {}[{}]", row_index, num_sampled_frames(),
                 filename_, game_index_);

  frame_index_t frame_index = get_frame_index(row_index);
  const GameLogCompactRecord* record = &get_record(get_mem_offset(frame_index));

  const GameLogCompactRecord* next_record = nullptr;
  if (frame_index + 1 < num_frames()) {
    next_record = &get_record(get_mem_offset(frame_index + 1));
  }

  int num_prev_frames_to_cp = std::min(InputEncoder::kNumFramesToEncode - 1, frame_index);
  int num_frames = num_prev_frames_to_cp + 1;

  InputFrame frames[InputEncoder::kNumFramesToEncode];

  for (int i = 0; i < num_prev_frames_to_cp; ++i) {
    int prev_frame_index = frame_index - num_prev_frames_to_cp + i;
    frames[i] = get_record(get_mem_offset(prev_frame_index)).frame;
  }
  frames[num_frames - 1] = record->frame;

  InputFrame final_frame = get_final_frame();

  InputEncoder input_encoder;
  input_encoder.restore(frames, num_frames);

  group::element_t sym = 0;
  if (apply_symmetry) {
    sym = input_encoder.get_random_symmetry();
    input_encoder.apply_symmetry(sym);
    Symmetries::apply(final_frame, sym);
  }

  typename GameLogView::Params view_params;
  view_params.record = record;
  view_params.next_record = next_record;
  view_params.cur_frame = &input_encoder.current_frame();
  view_params.final_frame = &final_frame;
  view_params.outcome = &get_outcome();
  view_params.sym = sym;

  GameLogView view(view_params);

  auto input = input_encoder.encode();

  constexpr int kInputSize = InputEncoder::Tensor::Dimensions::total_size;
  output_array = std::copy(input.data(), input.data() + kInputSize, output_array);

  constexpr size_t N = mp::Length_v<TrainingTargets>;
  for (int target_index : target_indices) {
    util::IndexedDispatcher<N>::call(target_index, [&](auto t) {
      using Target = mp::TypeAt_t<TrainingTargets, t>;
      using Tensor = Target::Tensor;
      constexpr int kSize = Tensor::Dimensions::total_size;

      Tensor tensor;
      tensor.setZero();
      bool mask = Target::encode(view, tensor);
      output_array = std::copy(tensor.data(), tensor.data() + kSize, output_array);
      output_array[0] = mask;
      output_array++;
    });
  }
}

template <::beta0::concepts::Spec Spec>
const typename GameReadLog<Spec>::InputFrame& GameReadLog<Spec>::get_final_frame() const {
  return *reinterpret_cast<const InputFrame*>(buffer_ + layout_.final_frame);
}

template <::beta0::concepts::Spec Spec>
const typename GameReadLog<Spec>::GameResultTensor& GameReadLog<Spec>::get_outcome() const {
  return *reinterpret_cast<const GameResultTensor*>(buffer_ + layout_.outcome);
}

template <::beta0::concepts::Spec Spec>
search::GameLogCommon::frame_index_t GameReadLog<Spec>::get_frame_index(int index) const {
  const frame_index_t* ptr = (const frame_index_t*)&buffer_[layout_.sampled_indices_start];
  return ptr[index];
}

template <::beta0::concepts::Spec Spec>
const typename GameReadLog<Spec>::GameLogCompactRecord& GameReadLog<Spec>::get_record(
  mem_offset_t mem_offset) const {
  const GameLogCompactRecord* ptr =
    (const GameLogCompactRecord*)&buffer_[layout_.records_start + mem_offset];
  return *ptr;
}

template <::beta0::concepts::Spec Spec>
typename GameReadLog<Spec>::mem_offset_t GameReadLog<Spec>::get_mem_offset(int frame_index) const {
  const mem_offset_t* mem_offsets_ptr = (const mem_offset_t*)&buffer_[layout_.mem_offsets_start];
  return mem_offsets_ptr[frame_index];
}

template <::beta0::concepts::Spec Spec>
GameWriteLog<Spec>::GameWriteLog(core::game_id_t id, int64_t start_timestamp)
    : search::GameWriteLogBase(id, start_timestamp) {}

template <::beta0::concepts::Spec Spec>
GameWriteLog<Spec>::~GameWriteLog() {
  for (GameLogFullRecord* full_record : full_records_) {
    delete full_record;
  }
}

template <::beta0::concepts::Spec Spec>
void GameWriteLog<Spec>::add(const TrainingInfo& training_info) {
  bool use_for_training = training_info.use_for_training;

  // TODO: get GameLogFullRecord objects from an object pool
  GameLogFullRecord* full_record = new GameLogFullRecord(training_info);
  full_records_.push_back(full_record);
  sample_count_ += use_for_training;
}

template <::beta0::concepts::Spec Spec>
void GameWriteLog<Spec>::add_terminal(const InputFrame& frame, const GameResultTensor& outcome) {
  terminal_added_ = true;
  final_frame_ = frame;
  outcome_ = outcome;

  int n = full_records_.size();
  if (n == 0) return;

  // Retroactively compute W_target for all positions via a lambda-discounted backward pass.
  //
  // S[t] represents the discounted future Q sum from position t onwards:
  //   S[T_max] = Q_root[T_max]
  //   S[t] = (1 - lambda) * Q_root[t+1] + lambda * S[t+1]
  //
  // W_target[t] = (Q_root[t] - S[t])^2   (element-wise, per-player)
  //
  // This follows the KataGo LoTV formulation with lambda = 5/6.

  ValueArray S = full_records_[n - 1]->Q_root;  // S[T_max] = Q_root[T_max]

  for (int t = n - 1; t >= 0; --t) {
    if (t < n - 1) {
      // S[t] = (1 - lambda) * Q_root[t+1] + lambda * S[t+1]
      //      = (1 - lambda) * Q_root[t+1] + lambda * prev_S
      // (prev_S holds S[t+1] from the previous iteration)
      S = (1.0f - kLambda) * full_records_[t + 1]->Q_root + kLambda * S;
    }

    ValueArray diff = full_records_[t]->Q_root - S;
    ValueArray W_target = diff * diff;  // element-wise square
    full_records_[t]->set_W_target(W_target);
  }
}

template <::beta0::concepts::Spec Spec>
bool GameWriteLog<Spec>::was_previous_entry_used_for_policy_training() const {
  if (full_records_.empty()) {
    return false;
  }
  return full_records_.back()->use_for_training;
}

template <::beta0::concepts::Spec Spec>
int GameWriteLog<Spec>::num_positions() const {
  return full_records_.size();
}

template <::beta0::concepts::Spec Spec>
bool GameWriteLog<Spec>::is_complete() const {
  return terminal_added_;
}

template <::beta0::concepts::Spec Spec>
bool GameWriteLog<Spec>::serialize_position(int move_num, std::vector<char>& data_buf) const {
  const GameLogFullRecord* full_record = full_records_[move_num];
  full_record->serialize(data_buf);
  return full_record->use_for_training;
}

template <::beta0::concepts::Spec Spec>
void GameWriteLog<Spec>::write_final_sections(std::vector<char>& buf) const {
  search::GameLogCommon::write_section(buf, &final_frame_);
  search::GameLogCommon::write_section(buf, &outcome_);
}

}  // namespace beta0
