#include "search/GameLog.hpp"

#include "util/Asserts.hpp"
#include "util/BitSet.hpp"
#include "util/IndexedDispatcher.hpp"
#include "util/MetaProgramming.hpp"

#include <algorithm>

namespace search {

template <search::concepts::Traits Traits>
GameReadLog<Traits>::DataLayout::DataLayout(const GameLogMetadata& m) {
  final_state = 0;
  outcome = align(final_state + sizeof(State));
  sampled_indices_start = align(outcome + sizeof(ValueTensor));
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
ShapeInfo* GameReadLog<Traits>::get_shape_info_array() {
  constexpr int n_primary_targets = mp::Length_v<PrimaryTargets>;
  constexpr int n_targets = mp::Length_v<AllTargets>;
  constexpr int n = n_targets + 2;  // 1 for input, 1 for terminator

  ShapeInfo* info_array = new ShapeInfo[n];
  info_array[0].template init<InputTensor>("input", -1, false);

  mp::constexpr_for<0, n_targets, 1>([&](auto a) {
    using Target = mp::TypeAt_t<AllTargets, a>;
    using Tensor = Target::Tensor;
    bool primary = (a < n_primary_targets);
    info_array[1 + a].template init<Tensor>(Target::kName, a, primary);
  });

  return info_array;
}

template <search::concepts::Traits Traits>
void GameReadLog<Traits>::load(int row_index, bool apply_symmetry,
                               const std::vector<int>& target_indices, float* output_array) const {
  // RELEASE_ASSERT(row_index >= 0 && row_index < num_sampled_positions(),
  //                "Index {} out of bounds [0, {}) in {}[{}]", row_index, num_sampled_positions(),
  //                filename_, game_index_);

  // pos_index_t state_index = get_pos_index(row_index);
  // mem_offset_t mem_offset = get_mem_offset(state_index);
  // const GameLogCompactRecord& record = get_record(mem_offset);

  // int num_prev_states_to_cp = std::min(Game::Constants::kNumPreviousStatesToEncode, state_index);
  // int num_states = num_prev_states_to_cp + 1;

  // State states[num_states];
  // for (int i = 0; i < num_prev_states_to_cp; ++i) {
  //   int prev_state_index = state_index - num_prev_states_to_cp + i;
  //   states[i] = get_record(get_mem_offset(prev_state_index)).position;
  // }
  // states[num_states - 1] = record.position;

  // State* start_pos = &states[0];
  // State* cur_pos = &states[num_states - 1];
  // State final_state = get_final_state();

  // group::element_t sym = 0;
  // if (apply_symmetry) {
  //   sym = bitset_util::choose_random_on_index(Game::Symmetries::get_mask(*cur_pos));
  // }

  // for (int i = 0; i < num_states; ++i) {
  //   Game::Symmetries::apply(states[i], sym);
  // }
  // Game::Symmetries::apply(final_state, sym);

  // float Q_prior = record.Q_prior;
  // float Q_posterior = record.Q_posterior;
  // core::seat_index_t active_seat = record.active_seat;
  // core::action_mode_t mode = record.action_mode;

  // PolicyTensor policy;
  // bool policy_valid = get_policy(mem_offset, policy);
  // if (policy_valid) Game::Symmetries::apply(policy, sym, mode);

  // ActionValueTensor action_values;
  // bool action_values_valid = get_action_values(mem_offset, action_values);
  // if (action_values_valid) Game::Symmetries::apply(action_values, sym, mode);

  // ActionValueTensor action_value_uncertainties;
  // bool action_value_uncertainties_valid =
  //   get_action_value_uncertainties(mem_offset, action_value_uncertainties);
  // if (action_value_uncertainties_valid)
  //   Game::Symmetries::apply(action_value_uncertainties, sym, mode);

  // PolicyTensor next_policy;
  // bool next_policy_valid = false;
  // bool has_next = state_index + 1 < num_positions();

  // if (has_next) {
  //   mem_offset_t next_mem_offset = get_mem_offset(state_index + 1);
  //   next_policy_valid = get_policy(next_mem_offset, next_policy);
  //   if (next_policy_valid) {
  //     Game::Symmetries::apply(next_policy, sym, get_record(next_mem_offset).action_mode);
  //   }
  // }

  // const ValueTensor& outcome = get_outcome();

  // constexpr int kInputSize = InputTensorizor::Tensor::Dimensions::total_size;
  // auto input = InputTensorizor::tensorize(start_pos, cur_pos);
  // output_array = std::copy(input.data(), input.data() + kInputSize, output_array);

  // PolicyTensor* policy_ptr = policy_valid ? &policy : nullptr;
  // PolicyTensor* next_policy_ptr = next_policy_valid ? &next_policy : nullptr;
  // ActionValueTensor* action_values_ptr = action_values_valid ? &action_values : nullptr;
  // ActionValueTensor* action_value_uncertainties_ptr =
  //   action_value_uncertainties_valid ? &action_value_uncertainties : nullptr;

  // GameLogView view(cur_pos, &final_state, &outcome, policy_ptr, next_policy_ptr,
  // action_values_ptr,
  //                  action_value_uncertainties_ptr, Q_prior, Q_posterior, active_seat);

  // constexpr size_t N = mp::Length_v<AllTargets>;
  // for (int target_index : target_indices) {
  //   util::IndexedDispatcher<N>::call(target_index, [&](auto t) {
  //     using Target = mp::TypeAt_t<AllTargets, t>;
  //     using Tensor = Target::Tensor;
  //     constexpr int kSize = Tensor::Dimensions::total_size;

  //     Tensor tensor;
  //     bool mask = Target::tensorize(view, tensor);
  //     output_array = std::copy(tensor.data(), tensor.data() + kSize, output_array);
  //     output_array[0] = mask;
  //     output_array++;
  //   });
  // }
}

// TODO: bring back replay()

// template <search::concepts::Traits Traits>
// void GameReadLog<Traits>::replay() const {
//   using Array = eigen_util::FArray<Game::Types::kMaxNumActions>;
//   int n = num_positions();
//   action_t last_action = -1;
//   for (int i = 0; i < n; ++i) {
//     mem_offset_t mem_offset = get_mem_offset(i);
//     const GameLogCompactRecord& record = get_record(mem_offset);

//     const State& pos = record.position;
//     seat_index_t active_seat = record.active_seat;
//     action_mode_t mode = record.action_mode;
//     action_t action = record.action;

//     Game::IO::print_state(std::cout, pos, last_action);
//     std::cout << "active seat: " << (int)active_seat << std::endl;

//     if (i < n - 1) {
//       PolicyTensor policy;
//       ActionValueTensor action_values_target;
//       bool policy_valid = get_policy(mem_offset, policy);
//       bool action_values_valid = get_action_values(mem_offset, action_values_target);

//       if (policy_valid || action_values_valid) {
//         Array action_arr(policy.size());
//         Array policy_arr(policy.size());
//         Array action_values_arr(action_values_target.size());
//         RELEASE_ASSERT(policy.size() == action_values_target.size());
//         for (action_t a = 0; a < policy.size(); ++a) {
//           action_arr(a) = a;
//           policy_arr(a) = policy(a);
//           action_values_arr(a) = action_values_target(a);
//         }

//         static std::vector<std::string> columns = {"action", "policy", "action_values"};
//         auto data = eigen_util::concatenate_columns(action_arr, policy_arr, action_values_arr);

//         eigen_util::PrintArrayFormatMap fmt_map{
//             {"action",
//              [&](float x) {
//                return std::string(x == action ? "*" : " ") + Game::IO::action_to_str(x, mode);
//              }},
//         };
//         eigen_util::print_array(std::cout, data, columns, &fmt_map);
//       }
//     }
//     std::cout << std::endl;
//     last_action = action;
//   }

//   Game::IO::print_state(std::cout, get_final_state(), last_action);
//   std::cout << "OUTCOME: " << std::endl;
//   std::cout << get_outcome() << std::endl;
// }

template <search::concepts::Traits Traits>
bool GameReadLog<Traits>::get_policy(mem_offset_t mem_offset, PolicyTensor& policy) const {
  int full_offset = layout_.records_start + mem_offset + sizeof(GameLogCompactRecord);
  const TensorData* policy_data = (const TensorData*)&buffer_[full_offset];

  return policy_data->load(policy);
}

template <search::concepts::Traits Traits>
bool GameReadLog<Traits>::get_action_values(mem_offset_t mem_offset,
                                            ActionValueTensor& action_values) const {
  int full_offset = layout_.records_start + mem_offset + sizeof(GameLogCompactRecord);

  const TensorData* policy_data = (const TensorData*)&buffer_[full_offset];
  full_offset += policy_data->size();

  const TensorData* action_values_data = (const TensorData*)&buffer_[full_offset];

  return action_values_data->load(action_values);
}

template <search::concepts::Traits Traits>
bool GameReadLog<Traits>::get_action_value_uncertainties(
  mem_offset_t mem_offset, ActionValueTensor& action_value_uncertainties) const {
  int full_offset = layout_.records_start + mem_offset + sizeof(GameLogCompactRecord);

  const TensorData* policy_data = (const TensorData*)&buffer_[full_offset];
  full_offset += policy_data->size();

  const TensorData* action_values_data = (const TensorData*)&buffer_[full_offset];
  full_offset += action_values_data->size();

  const TensorData* action_value_uncertainties_data = (const TensorData*)&buffer_[full_offset];

  return action_value_uncertainties_data->load(action_value_uncertainties);
}

template <search::concepts::Traits Traits>
const typename GameReadLog<Traits>::State& GameReadLog<Traits>::get_final_state() const {
  return *reinterpret_cast<const State*>(buffer_ + layout_.final_state);
}

template <search::concepts::Traits Traits>
const typename GameReadLog<Traits>::ValueTensor& GameReadLog<Traits>::get_outcome() const {
  return *reinterpret_cast<const ValueTensor*>(buffer_ + layout_.outcome);
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
void GameWriteLog<Traits>::add_terminal(const State& state, const ValueTensor& outcome) {
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
