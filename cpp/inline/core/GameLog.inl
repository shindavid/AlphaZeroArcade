#include <core/GameLog.hpp>

#include <util/BitSet.hpp>
#include <util/EigenUtil.hpp>
#include <util/LoggingUtil.hpp>
#include <util/Math.hpp>
#include <util/MetaProgramming.hpp>

#include <limits>

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

inline ShapeInfo::~ShapeInfo() {
  delete[] dims;
}

inline constexpr int GameLogBase::align(int offset) {
  return math::round_up_to_nearest_multiple(offset, kAlignment);
}

template <concepts::Game Game>
GameLog<Game>::TensorData::TensorData(bool valid, const PolicyTensor& tensor) {
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
int GameLog<Game>::TensorData::write_to(std::ostream& os) const {
  int s = size();
  os.write(reinterpret_cast<const char*>(this), s);
  return s;
}

template <concepts::Game Game>
bool GameLog<Game>::TensorData::load(PolicyTensor& tensor) const {
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
      sparse_tensor_entry_t s = data.sparse_repr.x[i];
      tensor(s.offset) = s.probability;
    }
  }
  return true;
}

template <concepts::Game Game>
GameLog<Game>::FileLayout::FileLayout(const Header& h) {
  header = 0;
  final_state = align(0 + sizeof(Header));
  outcome = align(final_state + sizeof(State));
  sampled_indices_start = align(outcome + sizeof(ValueTensor));
  mem_offsets_start = align(sampled_indices_start + sizeof(pos_index_t) * h.num_samples);
  records_start = align(mem_offsets_start + sizeof(mem_offset_t) * h.num_positions);
}

template <concepts::Game Game>
GameLog<Game>::GameLog(const char* filename)
    : filename_(filename),
      buffer_(get_buffer()),
      layout_(header()) {
  util::release_assert(num_positions() > 0, "Empty game log file: %s", filename_.c_str());
}

template <concepts::Game Game>
GameLog<Game>::~GameLog() {
  delete[] buffer_;
}

template <concepts::Game Game>
ShapeInfo* GameLog<Game>::get_shape_info_array() {
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

template <concepts::Game Game>
void GameLog<Game>::load(int index, bool apply_symmetry, float* input_values, int* target_indices,
                         float** target_arrays, bool** target_masks) const {
  util::release_assert(index >= 0 && index < num_sampled_positions(),
                       "Index %d out of bounds [0, %d) in %s", index, num_sampled_positions(),
                       filename_.c_str());

  pos_index_t state_index = get_pos_index(index);
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

  auto input = InputTensorizor::tensorize(start_pos, cur_pos);
  memcpy(input_values, input.data(), input.size() * sizeof(float));

  PolicyTensor* policy_ptr = policy_valid ? &policy : nullptr;
  PolicyTensor* next_policy_ptr = next_policy_valid ? &next_policy : nullptr;
  ActionValueTensor* action_values_ptr = action_values_valid ? &action_values : nullptr;
  GameLogView view{cur_pos, &final_state, &outcome, policy_ptr,
                   next_policy_ptr, action_values_ptr, active_seat};

  constexpr size_t N = mp::Length_v<TrainingTargetsList>;

  for (int t = 0;; ++t) {
    int target_index = target_indices[t];
    if (target_index < 0) break;

    mp::constexpr_for<0, N, 1>([&](auto a) {
      if (target_index == a) {
        using Target = mp::TypeAt_t<TrainingTargetsList, a>;
        using Tensor = Target::Tensor;
        Tensor tensor;
        target_masks[t][0] = Target::tensorize(view, tensor);
        memcpy(target_arrays[t], tensor.data(), tensor.size() * sizeof(float));
      }
    });
  }
}

template <concepts::Game Game>
void GameLog<Game>::replay() const {
  int n = num_positions();
  action_t last_action = -1;
  for (int i = 0; i < n; ++i) {
    mem_offset_t mem_offset = get_mem_offset(i);
    const Record& record = get_record(mem_offset);

    const State& pos = record.position;
    seat_index_t active_seat = record.active_seat;
    action_mode_t mode = record.action_mode;
    action_t action = record.action;

    Game::IO::print_state(std::cout, pos, last_action);
    std::cout << " seat: " << (int)active_seat << std::endl;

    if (i < n - 1) {
      ActionValueTensor action_values_target;
      bool action_values_valid = get_action_values(mem_offset, action_values_target);
      PolicyTensor policy;
      bool policy_valid = get_policy(mem_offset, policy);
      if (policy_valid || action_values_valid) {
        printf("  %3s  %8s %8s\n", "a", "policy", "AV");
        for (action_t a = 0; a < Game::Types::kMaxNumActions; ++a) {
          if (policy(a) > 0) {
            char p = a == action ? '*' : ' ';
            std::string s = Game::IO::action_to_str(a, mode);
            printf("%c %3s: %.6f %.6f\n", p, s.c_str(), policy(a), action_values_target(a));
          }
        }
      }
    }
    std::cout << std::endl;
    last_action = action;
  }

  Game::IO::print_state(std::cout, get_final_state(), last_action);
  std::cout << "OUTCOME: " << std::endl;
  std::cout << get_outcome() << std::endl;
}

template <concepts::Game Game>
char* GameLog<Game>::get_buffer() const {
  FILE* file = fopen(filename_.c_str(), "rb");
  util::clean_assert(file, "Failed to open file '%s'", filename_.c_str());

  // get file size:
  fseek(file, 0, SEEK_END);
  int file_size = ftell(file);

  char* buffer = new char[file_size];
  fseek(file, 0, SEEK_SET);
  int read_size = fread(buffer, 1, file_size, file);
  if (read_size != file_size) {
    fclose(file);
    delete[] buffer;
    throw util::Exception("Failed to read file '%s' (%d!=%d)", filename_.c_str(), read_size,
                          file_size);
  }
  fclose(file);

  return buffer;
}

template <concepts::Game Game>
const GameLogBase::Header& GameLog<Game>::header() const {
  return *reinterpret_cast<const GameLogBase::Header*>(buffer_);
}

template <concepts::Game Game>
bool GameLog<Game>::get_policy(mem_offset_t mem_offset, PolicyTensor& policy) const {
  int full_offset = layout_.records_start + mem_offset + sizeof(Record);
  const TensorData* policy_data = (const TensorData*) &buffer_[full_offset];

  return policy_data->load(policy);
}

template <concepts::Game Game>
bool GameLog<Game>::get_action_values(mem_offset_t mem_offset,
                                      ActionValueTensor& action_values) const {
  int full_offset = layout_.records_start + mem_offset + sizeof(Record);
  const TensorData* policy_data = (const TensorData*)&buffer_[full_offset];
  full_offset += policy_data->size();
  const TensorData* action_values_data = (const TensorData*)&buffer_[full_offset];

  return action_values_data->load(action_values);
}

template <concepts::Game Game>
const typename GameLog<Game>::State& GameLog<Game>::get_final_state() const {
  return *reinterpret_cast<State*>(buffer_ + layout_.final_state);
}

template <concepts::Game Game>
const typename GameLog<Game>::ValueTensor& GameLog<Game>::get_outcome() const {
  return *reinterpret_cast<ValueTensor*>(buffer_ + layout_.outcome);
}

template <concepts::Game Game>
GameLogBase::pos_index_t GameLog<Game>::get_pos_index(int index) const {
  pos_index_t* ptr = (pos_index_t*) &buffer_[layout_.sampled_indices_start];
  return ptr[index];
}

template <concepts::Game Game>
const typename GameLog<Game>::Record& GameLog<Game>::get_record(mem_offset_t mem_offset) const {
  const Record* ptr = (const Record*) &buffer_[layout_.records_start + mem_offset];
  return *ptr;
}

template <concepts::Game Game>
typename GameLog<Game>::mem_offset_t GameLog<Game>::get_mem_offset(int state_index) const {
  mem_offset_t* mem_offsets_ptr = (mem_offset_t*) &buffer_[layout_.mem_offsets_start];
  return mem_offsets_ptr[state_index];
}

template <concepts::Game Game>
GameLogWriter<Game>::GameLogWriter(game_id_t id, int64_t start_timestamp)
    : id_(id), start_timestamp_(start_timestamp) {}

template <concepts::Game Game>
GameLogWriter<Game>::~GameLogWriter() {
  for (Entry* entry : entries_) {
    delete entry;
  }
}

template <concepts::Game Game>
void GameLogWriter<Game>::add(const State& state, action_t action, seat_index_t active_seat,
                              const PolicyTensor* policy_target,
                              const ActionValueTensor* action_values, bool use_for_training) {
  // TODO: get entries from a thread-specific object pool
  Entry* entry = new Entry();
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
void GameLogWriter<Game>::add_terminal(const State& state, const ValueTensor& outcome) {
  util::release_assert(!terminal_added_);
  terminal_added_ = true;
  final_state_ = state;
  outcome_ = outcome;
}

template <concepts::Game Game>
void GameLogWriter<Game>::serialize(std::ostream& stream) const {
  util::release_assert(terminal_added_);
  int num_entries = entries_.size();

  std::vector<pos_index_t> sampled_indices;
  std::vector<mem_offset_t> mem_offsets;
  std::ostringstream data_stream;

  sampled_indices.reserve(sample_count_);
  mem_offsets.reserve(num_entries);

  mem_offset_t mem_offset = 0;

  for (int move_num = 0; move_num < num_entries; ++move_num) {
    const Entry* entry = entries_[move_num];

    mem_offsets.push_back(mem_offset);
    if (entry->use_for_training) {
      sampled_indices.push_back(move_num);
    }

    Record record;
    record.position = entry->position;
    record.active_seat = entry->active_seat;
    record.action_mode = Game::Rules::get_action_mode(entry->position);
    record.action = entry->action;

    TensorData policy(entry->policy_target_is_valid, entry->policy_target);
    TensorData action_values(entry->action_values_are_valid, entry->action_values);

    mem_offset += write_section(data_stream, &record, 1, false);
    mem_offset += policy.write_to(data_stream);
    mem_offset += action_values.write_to(data_stream);
  }

  Header header;
  header.num_samples = sample_count_;
  header.num_positions = num_entries;
  header.extra = 0;

  write_section(stream, &header);
  write_section(stream, &final_state_);
  write_section(stream, &outcome_);
  write_section(stream, sampled_indices.data(), sampled_indices.size());
  write_section(stream, mem_offsets.data(), mem_offsets.size());
  stream << data_stream.str();
}

template <concepts::Game Game>
bool GameLogWriter<Game>::was_previous_entry_used_for_policy_training() const {
  if (entries_.empty()) {
    return false;
  }
  return entries_.back()->use_for_training;
}

template <concepts::Game Game>
template <typename T>
int GameLogWriter<Game>::write_section(std::ostream& os, const T* t, int count, bool pad) {
  constexpr int A = GameLogBase::kAlignment;
  int n_bytes = sizeof(T) * count;
  os.write(reinterpret_cast<const char*>(t), n_bytes);

  if (!pad) return n_bytes;

  int remainder = n_bytes % A;
  int padding = 0;
  if (remainder) {
    padding = A - remainder;
    uint8_t zeroes[A] = {0};
    os.write(reinterpret_cast<const char*>(zeroes), padding);
  }
  return n_bytes + padding;
}

}  // namespace core
