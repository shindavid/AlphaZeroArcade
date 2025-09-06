#include "core/GameLog.hpp"

#include "util/Asserts.hpp"
#include "util/BitSet.hpp"
#include "util/EigenUtil.hpp"
#include "util/FileUtil.hpp"
#include "util/IndexedDispatcher.hpp"
#include "util/Math.hpp"
#include "util/MetaProgramming.hpp"

#include <algorithm>

namespace core {

template <eigen_util::concepts::FTensor Tensor>
void ShapeInfo::init(const char* nm, int target_idx) {
  using Shape = Tensor::Dimensions;
  this->name = nm;
  this->dims = new int[Shape::count];
  this->num_dims = Shape::count;
  this->target_index = target_idx;

  Shape shape;
  for (int i = 0; i < Shape::count; ++i) {
    this->dims[i] = shape[i];
  }
}

inline ShapeInfo::~ShapeInfo() { delete[] dims; }

inline GameLogFileReader::GameLogFileReader(const char* buffer) : buffer_(buffer) {}

inline const GameLogFileHeader& GameLogFileReader::header() const {
  return *reinterpret_cast<const GameLogFileHeader*>(buffer_);
}

inline int GameLogFileReader::num_games() const { return header().num_games; }

inline int GameLogFileReader::num_samples(int game_index) const {
  return metadata(game_index).num_samples;
}

inline const char* GameLogFileReader::game_data_buffer(int game_index) const {
  const GameLogMetadata& m = metadata(game_index);
  return buffer_ + m.start_offset;
}

inline const GameLogMetadata& GameLogFileReader::metadata(int game_index) const {
  const GameLogFileHeader& h = header();
  constexpr auto m = sizeof(GameLogMetadata);
  return *reinterpret_cast<const GameLogMetadata*>(buffer_ + sizeof(h) + game_index * m);
}

inline void GameLogCommon::merge_files(const char** input_filenames, int n_input_filenames,
                                       const char* output_filename) {
  std::vector<GameLogFileReader> readers;

  for (int i = 0; i < n_input_filenames; ++i) {
    char* buffer = util::read_file(input_filenames[i]);
    readers.emplace_back(buffer);
  }

  GameLogFileHeader header;
  for (const auto& reader : readers) {
    header.num_games += reader.num_games();
    header.num_rows += reader.header().num_rows;
  }

  struct game_t {
    int reader_index;
    int game_index;
  };

  uint64_t total_data_size = 0;
  std::vector<game_t> games;
  games.reserve(header.num_games);
  for (int i = 0; i < n_input_filenames; ++i) {
    int num_games = readers[i].num_games();
    for (int j = 0; j < num_games; ++j) {
      games.push_back(game_t{i, j});
      total_data_size += readers[i].metadata(j).data_size;
    }
  }

  std::sort(games.begin(), games.end(), [&](const game_t& a, const game_t& b) {
    return readers[a.reader_index].metadata(a.game_index).start_timestamp <
           readers[b.reader_index].metadata(b.game_index).start_timestamp;
  });

  uint32_t start_offset = sizeof(GameLogFileHeader) + sizeof(GameLogMetadata) * header.num_games;
  size_t out_buffer_size = start_offset + total_data_size;

  std::vector<char> out_buffer;
  out_buffer.reserve(out_buffer_size);

  const char* h = reinterpret_cast<const char*>(&header);
  out_buffer.insert(out_buffer.end(), h, h + sizeof(header));

  for (const game_t& game : games) {
    const GameLogFileReader& reader = readers[game.reader_index];
    GameLogMetadata metadata = reader.metadata(game.game_index);

    metadata.start_offset = start_offset;
    start_offset += metadata.data_size;

    const char* m = reinterpret_cast<const char*>(&metadata);
    out_buffer.insert(out_buffer.end(), m, m + sizeof(metadata));
  }

  for (const game_t& game : games) {
    const GameLogFileReader& reader = readers[game.reader_index];
    GameLogMetadata metadata = reader.metadata(game.game_index);
    const char* game_buffer = reader.game_data_buffer(game.game_index);
    out_buffer.insert(out_buffer.end(), game_buffer, game_buffer + metadata.data_size);
  }

  RELEASE_ASSERT(out_buffer.size() == out_buffer_size, "Size mismatch {} != {}", out_buffer.size(),
                 out_buffer_size);

  std::ofstream output_file(output_filename, std::ios::binary);
  output_file.write(out_buffer.data(), out_buffer.size());
  RELEASE_ASSERT(output_file.good(), "Failed to write to {}", output_filename);
  output_file.close();

  for (const auto& reader : readers) {
    delete[] reader.buffer();
  }
}

inline constexpr int GameLogCommon::align(int offset) {
  return math::round_up_to_nearest_multiple(offset, kAlignment);
}

template <typename T>
int GameLogCommon::write_section(std::vector<char>& buf, const T* t, int count, bool pad) {
  constexpr int A = kAlignment;
  int n_bytes = sizeof(T) * count;
  const char* bytes = reinterpret_cast<const char*>(t);
  buf.insert(buf.end(), bytes, bytes + n_bytes);

  if (!pad) return n_bytes;

  int remainder = n_bytes % A;
  int padding = 0;
  if (remainder) {
    padding = A - remainder;
    uint8_t zeroes[A] = {0};
    buf.insert(buf.end(), zeroes, zeroes + padding);
  }
  return n_bytes + padding;
}

template <concepts::Game Game>
GameLogBase<Game>::TensorData::TensorData(bool valid, const PolicyTensor& tensor) {
  if (!valid) {
    encoding = 0;
    return;
  }

  constexpr int N = PolicyTensor::Dimensions::total_size;
  int num_nonzero_entries = eigen_util::count(tensor);

  const auto* src = tensor.data();
  if (num_nonzero_entries <= kSparseCapacity) {
    int n = num_nonzero_entries;
    encoding = 2 * n;
    auto* dst = data.sparse_repr.x;

    int w = 0;
    for (int r = 0; r < N; ++r) {
      if (src[r]) {
        dst[w].offset = r;
        dst[w].probability = src[r];
        ++w;
      }
    }
  } else {
    encoding = -N;
    auto* dst = data.dense_repr.x;
    std::copy(src, src + N, dst);
  }
}

template <concepts::Game Game>
int GameLogBase<Game>::TensorData::write_to(std::vector<char>& buf) const {
  int s = size();
  const char* bytes = reinterpret_cast<const char*>(this);
  buf.insert(buf.end(), bytes, bytes + s);
  return s;
}

template <concepts::Game Game>
bool GameLogBase<Game>::TensorData::load(PolicyTensor& tensor) const {
  if (encoding == 0) {
    // no policy target
    return false;
  }

  if (encoding < 0) {
    // dense format
    int n = -encoding;
    const float* src = &data.dense_repr.x[0];
    float* dst = tensor.data();
    std::copy(src, src + n, dst);
  } else {
    // sparse format
    int n = encoding / 2;
    tensor.setZero();

    for (int i = 0; i < n; ++i) {
      SparseTensorEntry s = data.sparse_repr.x[i];
      tensor(s.offset) = s.probability;
    }
  }
  return true;
}

template <concepts::EvalSpec EvalSpec>
GameReadLog<EvalSpec>::DataLayout::DataLayout(const GameLogMetadata& m) {
  final_state = 0;
  outcome = align(final_state + sizeof(State));
  sampled_indices_start = align(outcome + sizeof(ValueTensor));
  mem_offsets_start = align(sampled_indices_start + sizeof(pos_index_t) * m.num_samples);
  records_start = align(mem_offsets_start + sizeof(mem_offset_t) * m.num_positions);
}

template <concepts::EvalSpec EvalSpec>
GameReadLog<EvalSpec>::GameReadLog(const char* filename, int game_index,
                                   const GameLogMetadata& metadata, const char* buffer)
    : filename_(filename),
      game_index_(game_index),
      metadata_(metadata),
      buffer_(buffer),
      layout_(metadata) {
  RELEASE_ASSERT(num_positions() > 0, "Empty game log file {}[{}]", filename, game_index);
}

template <concepts::EvalSpec EvalSpec>
ShapeInfo* GameReadLog<EvalSpec>::get_shape_info_array() {
  constexpr int n_targets = mp::Length_v<TrainingTargetsList>;
  constexpr int n = n_targets + 2;  // 1 for input, 1 for terminator

  ShapeInfo* info_array = new ShapeInfo[n];
  info_array[0].template init<InputTensor>("input", -1);

  mp::constexpr_for<0, n_targets, 1>([&](auto a) {
    using Target = mp::TypeAt_t<TrainingTargetsList, a>;
    using Tensor = Target::Tensor;
    info_array[1 + a].template init<Tensor>(Target::kName, a);
  });

  return info_array;
}

template <concepts::EvalSpec EvalSpec>
void GameReadLog<EvalSpec>::load(int row_index, bool apply_symmetry,
                                 const std::vector<int>& target_indices,
                                 float* output_array) const {
  RELEASE_ASSERT(row_index >= 0 && row_index < num_sampled_positions(),
                 "Index {} out of bounds [0, {}) in {}[{}]", row_index, num_sampled_positions(),
                 filename_, game_index_);

  pos_index_t state_index = get_pos_index(row_index);
  mem_offset_t mem_offset = get_mem_offset(state_index);
  const Record& record = get_record(mem_offset);

  int num_prev_states_to_cp = std::min(Game::Constants::kNumPreviousStatesToEncode, state_index);
  int num_states = num_prev_states_to_cp + 1;

  State states[num_states];
  for (int i = 0; i < num_prev_states_to_cp; ++i) {
    int prev_state_index = state_index - num_prev_states_to_cp + i;
    states[i] = get_record(get_mem_offset(prev_state_index)).position;
  }
  states[num_states - 1] = record.position;

  State* start_pos = &states[0];
  State* cur_pos = &states[num_states - 1];
  State final_state = get_final_state();

  group::element_t sym = 0;
  if (apply_symmetry) {
    sym = bitset_util::choose_random_on_index(Game::Symmetries::get_mask(*cur_pos));
  }

  for (int i = 0; i < num_states; ++i) {
    Game::Symmetries::apply(states[i], sym);
  }
  Game::Symmetries::apply(final_state, sym);

  seat_index_t active_seat = record.active_seat;
  action_mode_t mode = record.action_mode;

  PolicyTensor policy;
  bool policy_valid = get_policy(mem_offset, policy);
  if (policy_valid) Game::Symmetries::apply(policy, sym, mode);

  ActionValueTensor action_values;
  bool action_values_valid = get_action_values(mem_offset, action_values);
  if (action_values_valid) Game::Symmetries::apply(action_values, sym, mode);

  PolicyTensor next_policy;
  bool next_policy_valid = false;
  bool has_next = state_index + 1 < num_positions();

  if (has_next) {
    mem_offset_t next_mem_offset = get_mem_offset(state_index + 1);
    next_policy_valid = get_policy(next_mem_offset, next_policy);
    if (next_policy_valid) {
      Game::Symmetries::apply(next_policy, sym, get_record(next_mem_offset).action_mode);
    }
  }

  const ValueTensor& outcome = get_outcome();

  constexpr int kInputSize = InputTensorizor::Tensor::Dimensions::total_size;
  auto input = InputTensorizor::tensorize(start_pos, cur_pos);
  output_array = std::copy(input.data(), input.data() + kInputSize, output_array);

  PolicyTensor* policy_ptr = policy_valid ? &policy : nullptr;
  PolicyTensor* next_policy_ptr = next_policy_valid ? &next_policy : nullptr;
  ActionValueTensor* action_values_ptr = action_values_valid ? &action_values : nullptr;
  GameLogView view{cur_pos,         &final_state,      &outcome,   policy_ptr,
                   next_policy_ptr, action_values_ptr, active_seat};

  constexpr size_t N = mp::Length_v<TrainingTargetsList>;
  for (int target_index : target_indices) {
    util::IndexedDispatcher<N>::call(target_index, [&](auto t) {
      using Target = mp::TypeAt_t<TrainingTargetsList, t>;
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

// TODO: bring back replay()

// template <concepts::EvalSpec EvalSpec>
// void GameReadLog<EvalSpec>::replay() const {
//   using Array = eigen_util::FArray<Game::Types::kMaxNumActions>;
//   int n = num_positions();
//   action_t last_action = -1;
//   for (int i = 0; i < n; ++i) {
//     mem_offset_t mem_offset = get_mem_offset(i);
//     const Record& record = get_record(mem_offset);

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

template <concepts::EvalSpec EvalSpec>
bool GameReadLog<EvalSpec>::get_policy(mem_offset_t mem_offset, PolicyTensor& policy) const {
  int full_offset = layout_.records_start + mem_offset + sizeof(Record);
  const TensorData* policy_data = (const TensorData*)&buffer_[full_offset];

  return policy_data->load(policy);
}

template <concepts::EvalSpec EvalSpec>
bool GameReadLog<EvalSpec>::get_action_values(mem_offset_t mem_offset,
                                              ActionValueTensor& action_values) const {
  int full_offset = layout_.records_start + mem_offset + sizeof(Record);
  const TensorData* policy_data = (const TensorData*)&buffer_[full_offset];
  full_offset += policy_data->size();
  const TensorData* action_values_data = (const TensorData*)&buffer_[full_offset];

  return action_values_data->load(action_values);
}

template <concepts::EvalSpec EvalSpec>
const typename GameReadLog<EvalSpec>::State& GameReadLog<EvalSpec>::get_final_state() const {
  return *reinterpret_cast<const State*>(buffer_ + layout_.final_state);
}

template <concepts::EvalSpec EvalSpec>
const typename GameReadLog<EvalSpec>::ValueTensor& GameReadLog<EvalSpec>::get_outcome() const {
  return *reinterpret_cast<const ValueTensor*>(buffer_ + layout_.outcome);
}

template <concepts::EvalSpec EvalSpec>
GameLogCommon::pos_index_t GameReadLog<EvalSpec>::get_pos_index(int index) const {
  const pos_index_t* ptr = (const pos_index_t*)&buffer_[layout_.sampled_indices_start];
  return ptr[index];
}

template <concepts::EvalSpec EvalSpec>
const typename GameReadLog<EvalSpec>::Record& GameReadLog<EvalSpec>::get_record(
  mem_offset_t mem_offset) const {
  const Record* ptr = (const Record*)&buffer_[layout_.records_start + mem_offset];
  return *ptr;
}

template <concepts::EvalSpec EvalSpec>
typename GameReadLog<EvalSpec>::mem_offset_t GameReadLog<EvalSpec>::get_mem_offset(
  int state_index) const {
  const mem_offset_t* mem_offsets_ptr = (const mem_offset_t*)&buffer_[layout_.mem_offsets_start];
  return mem_offsets_ptr[state_index];
}

template <concepts::Game Game>
GameWriteLog<Game>::GameWriteLog(game_id_t id, int64_t start_timestamp)
    : id_(id), start_timestamp_(start_timestamp) {}

template <concepts::Game Game>
GameWriteLog<Game>::~GameWriteLog() {
  for (WriteEntry* entry : entries_) {
    delete entry;
  }
}

template <concepts::Game Game>
void GameWriteLog<Game>::add(const State& state, action_t action, seat_index_t active_seat,
                             const PolicyTensor* policy_target,
                             const ActionValueTensor* action_values, bool use_for_training) {
  // TODO: get entries from a thread-specific object pool
  WriteEntry* entry = new WriteEntry();
  entry->position = state;
  if (policy_target) {
    entry->policy_target = *policy_target;
  } else {
    entry->policy_target.setZero();
  }
  if (action_values) {
    entry->action_values = *action_values;
  } else {
    entry->action_values.setZero();
  }
  entry->action = action;
  entry->active_seat = active_seat;
  entry->use_for_training = use_for_training;
  entry->policy_target_is_valid = policy_target != nullptr;
  entry->action_values_are_valid = action_values != nullptr;
  entries_.push_back(entry);
  sample_count_ += use_for_training;
}

template <concepts::Game Game>
void GameWriteLog<Game>::add_terminal(const State& state, const ValueTensor& outcome) {
  RELEASE_ASSERT(!terminal_added_);
  terminal_added_ = true;
  final_state_ = state;
  outcome_ = outcome;
}

template <concepts::Game Game>
bool GameWriteLog<Game>::was_previous_entry_used_for_policy_training() const {
  if (entries_.empty()) {
    return false;
  }
  return entries_.back()->use_for_training;
}

template <concepts::Game Game>
GameLogMetadata GameLogSerializer<Game>::serialize(const GameWriteLog* log, std::vector<char>& buf,
                                                   int client_id) {
  uint32_t start_buf_size = buf.size();
  RELEASE_ASSERT(log->terminal_added_);
  int num_entries = log->entries_.size();

  mem_offset_t mem_offset = 0;

  for (int move_num = 0; move_num < num_entries; ++move_num) {
    const WriteEntry* entry = log->entries_[move_num];

    mem_offsets_.push_back(mem_offset);
    if (entry->use_for_training) {
      sampled_indices_.push_back(move_num);
    }

    Record record;
    record.position = entry->position;
    record.active_seat = entry->active_seat;
    record.action_mode = Game::Rules::get_action_mode(entry->position);
    record.action = entry->action;

    TensorData policy(entry->policy_target_is_valid, entry->policy_target);
    TensorData action_values(entry->action_values_are_valid, entry->action_values);

    mem_offset += GameLogCommon::write_section(data_buf_, &record, 1, false);
    mem_offset += policy.write_to(data_buf_);
    mem_offset += action_values.write_to(data_buf_);
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
  metadata.num_positions = num_entries;
  metadata.client_id = client_id;

  return metadata;
}

}  // namespace core
