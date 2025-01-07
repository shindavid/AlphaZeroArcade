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
GameLog<Game>::GameLog(const char* filename)
    : filename_(filename),
      buffer_(get_buffer()),
      action_start_mem_offset_(action_start_mem_offset()),
      seat_index_start_mem_offset_(seat_index_start_mem_offset()),
      policy_target_index_start_mem_offset_(policy_target_index_start_mem_offset()),
      action_values_target_index_start_mem_offset_(action_values_target_index_start_mem_offset()),
      state_start_mem_offset_(state_start_mem_offset()),
      dense_policy_start_mem_offset_(dense_policy_start_mem_offset()),
      sparse_policy_entry_start_mem_offset_(sparse_policy_entry_start_mem_offset()),
      dense_action_values_start_mem_offset_(dense_action_values_start_mem_offset()),
      sparse_action_values_entry_start_mem_offset_(sparse_action_values_entry_start_mem_offset()) {
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
  bool has_next = state_index + 1 < num_non_terminal_positions();

  seat_index_t active_seat = get_active_seat(state_index);

  PolicyTensor policy, next_policy;
  bool policy_valid = get_policy(state_index, policy);
  bool next_policy_valid = has_next && get_policy(state_index + 1, next_policy);

  ActionValueTensor action_values;
  bool action_values_valid = get_action_values(state_index, action_values);

  int num_states_to_cp = 1 + std::min(Game::Constants::kNumPreviousStatesToEncode, state_index);
  int num_bytes_to_cp = num_states_to_cp * sizeof(State);

  State states[num_states_to_cp];
  std::memcpy(&states[0], get_state(state_index - num_states_to_cp + 1), num_bytes_to_cp);

  State* start_pos = &states[0];
  State* cur_pos = &states[num_states_to_cp - 1];
  State final_state = *get_state(num_positions() - 1);

  group::element_t sym = 0;
  if (apply_symmetry) {
    sym = bitset_util::choose_random_on_index(Game::Symmetries::get_mask(*cur_pos));
  }

  action_mode_t mode = Game::Rules::get_action_mode(*cur_pos);

  for (int i = 0; i < num_states_to_cp; ++i) {
    Game::Symmetries::apply(states[i], sym);
  }
  Game::Symmetries::apply(final_state, sym);
  Game::Symmetries::apply(policy, sym, mode);
  if (has_next) {
    action_mode_t next_mode = Game::Rules::get_action_mode(*get_state(state_index + 1));
    Game::Symmetries::apply(next_policy, sym, next_mode);
  }
  Game::Symmetries::apply(action_values, sym, mode);

  ValueTensor outcome = get_outcome();

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
  for (int i = 0; i < n; ++i) {
    const State* pos = get_state(i);
    action_mode_t mode = Game::Rules::get_action_mode(*pos);
    seat_index_t active_seat = get_active_seat(i);
    action_t last_action = get_prev_action(i);
    Game::IO::print_state(std::cout, *pos, last_action);
    std::cout << "active seat: " << (int)active_seat << std::endl;
    if (i < n - 1) {
      action_t action = get_prev_action(i + 1);
      PolicyTensor policy;
      bool policy_valid = get_policy(i, policy);
      if (!policy_valid) continue;

      bool add_newline = false;
      for (action_t a = 0; a < Game::Types::kMaxNumActions; ++a) {
        if (policy(a) > 0) {
          char p = a == action ? '*' : ' ';
          std::string s = Game::IO::action_to_str(a, mode);
          printf("%c %s: %.6f\n", p, s.c_str(), policy(a));
          add_newline = true;
        }
      }
      if (add_newline) std::cout << std::endl;
    }
  }
  std::cout << "OUTCOME: " << std::endl;
  std::cout << get_outcome() << std::endl;
}

template <concepts::Game Game>
int GameLog<Game>::num_sampled_positions() const {
  return header().num_samples;
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
GameLogBase::Header& GameLog<Game>::header() {
  return *reinterpret_cast<GameLogBase::Header*>(buffer_ + header_start_mem_offset());
}

template <concepts::Game Game>
const GameLogBase::Header& GameLog<Game>::header() const {
  return *reinterpret_cast<const GameLogBase::Header*>(buffer_ + header_start_mem_offset());
}

template <concepts::Game Game>
int GameLog<Game>::num_positions() const {
  return header().num_positions;
}

template <concepts::Game Game>
int GameLog<Game>::num_non_terminal_positions() const {
  return header().num_positions - 1;
}

template <concepts::Game Game>
int GameLog<Game>::num_dense_policies() const {
  return header().num_dense_policies;
}

template <concepts::Game Game>
int GameLog<Game>::num_sparse_policy_entries() const {
  return header().num_sparse_policy_entries;
}

template <concepts::Game Game>
int GameLog<Game>::num_dense_action_values() const {
  return header().num_dense_action_values;
}

template <concepts::Game Game>
int GameLog<Game>::num_sparse_action_values_entries() const {
  return header().num_sparse_action_values_entries;
}

template <concepts::Game Game>
constexpr int GameLog<Game>::header_start_mem_offset() {
  return 0;
}

template <concepts::Game Game>
constexpr int GameLog<Game>::outcome_start_mem_offset() {
  return header_start_mem_offset() + align(sizeof(Header));
}

template <concepts::Game Game>
constexpr int GameLog<Game>::sampled_indices_start_mem_offset() {
  return outcome_start_mem_offset() + align(sizeof(ValueTensor));
}

template <concepts::Game Game>
int GameLog<Game>::action_start_mem_offset() const {
  return sampled_indices_start_mem_offset() + align(num_sampled_positions() * sizeof(pos_index_t));
}

template <concepts::Game Game>
int GameLog<Game>::seat_index_start_mem_offset() const {
  return action_start_mem_offset_ + align(num_non_terminal_positions() * sizeof(action_t));
}

template <concepts::Game Game>
int GameLog<Game>::policy_target_index_start_mem_offset() const {
  return seat_index_start_mem_offset_ + align(num_non_terminal_positions() * sizeof(seat_index_t));
}

template <concepts::Game Game>
int GameLog<Game>::action_values_target_index_start_mem_offset() const {
  return policy_target_index_start_mem_offset_ +
         align(num_non_terminal_positions() * sizeof(tensor_index_t));
}

template <concepts::Game Game>
int GameLog<Game>::state_start_mem_offset() const {
  return action_values_target_index_start_mem_offset_ +
         align(num_non_terminal_positions() * sizeof(tensor_index_t));
}

template <concepts::Game Game>
int GameLog<Game>::dense_policy_start_mem_offset() const {
  return state_start_mem_offset_ + align(num_positions() * sizeof(State));
}

template <concepts::Game Game>
int GameLog<Game>::sparse_policy_entry_start_mem_offset() const {
  return dense_policy_start_mem_offset_ + align(num_dense_policies() * sizeof(PolicyTensor));
}

template <concepts::Game Game>
int GameLog<Game>::dense_action_values_start_mem_offset() const {
  return sparse_policy_entry_start_mem_offset_ +
         align(num_sparse_policy_entries() * sizeof(sparse_tensor_entry_t));
}

template <concepts::Game Game>
int GameLog<Game>::sparse_action_values_entry_start_mem_offset() const {
  return dense_action_values_start_mem_offset_ +
         align(num_dense_action_values() * sizeof(ActionValueTensor));
}

template <concepts::Game Game>
const action_t* GameLog<Game>::action_start_ptr() const {
  return reinterpret_cast<action_t*>(buffer_ + action_start_mem_offset_);
}

template <concepts::Game Game>
const seat_index_t* GameLog<Game>::seat_index_start_ptr() const {
  return reinterpret_cast<seat_index_t*>(buffer_ + seat_index_start_mem_offset_);
}

template <concepts::Game Game>
const GameLogBase::tensor_index_t* GameLog<Game>::policy_target_index_start_ptr() const {
  return reinterpret_cast<tensor_index_t*>(buffer_ + policy_target_index_start_mem_offset_);
}

template <concepts::Game Game>
const GameLogBase::tensor_index_t* GameLog<Game>::action_values_target_index_start_ptr() const {
  return reinterpret_cast<tensor_index_t*>(buffer_ + action_values_target_index_start_mem_offset_);
}

template <concepts::Game Game>
const typename GameLog<Game>::State* GameLog<Game>::state_start_ptr() const {
  return reinterpret_cast<State*>(buffer_ + state_start_mem_offset_);
}

template <concepts::Game Game>
const typename GameLog<Game>::PolicyTensor* GameLog<Game>::dense_policy_start_ptr() const {
  return reinterpret_cast<PolicyTensor*>(buffer_ + dense_policy_start_mem_offset_);
}

template <concepts::Game Game>
const GameLogBase::sparse_tensor_entry_t* GameLog<Game>::sparse_policy_entry_start_ptr() const {
  return reinterpret_cast<sparse_tensor_entry_t*>(buffer_ + sparse_policy_entry_start_mem_offset_);
}

template <concepts::Game Game>
const typename GameLog<Game>::ActionValueTensor* GameLog<Game>::dense_action_values_start_ptr()
    const {
  return reinterpret_cast<ActionValueTensor*>(buffer_ + dense_action_values_start_mem_offset_);
}

template <concepts::Game Game>
const GameLogBase::sparse_tensor_entry_t* GameLog<Game>::sparse_action_values_entry_start_ptr()
    const {
  return reinterpret_cast<sparse_tensor_entry_t*>(buffer_ +
                                                  sparse_action_values_entry_start_mem_offset_);
}

template <concepts::Game Game>
const GameLogBase::pos_index_t* GameLog<Game>::sampled_indices_start_ptr() const {
  return reinterpret_cast<pos_index_t*>(buffer_ + sampled_indices_start_mem_offset());
}

template <concepts::Game Game>
seat_index_t GameLog<Game>::get_active_seat(int state_index) const {
  return seat_index_start_ptr()[state_index];
}

template <concepts::Game Game>
bool GameLog<Game>::get_policy(int state_index, PolicyTensor& policy) const {
  tensor_index_t index = policy_target_index_start_ptr()[state_index];

  if (index.start < index.end) {
    // sparse case
    policy.setZero();
    const sparse_tensor_entry_t* sparse_start = sparse_policy_entry_start_ptr();
    for (int i = index.start; i < index.end; ++i) {
      sparse_tensor_entry_t entry = sparse_start[i];
      policy(entry.offset) = entry.probability;
    }
    return true;
  } else if (index.start == index.end) {
    if (index.start < 0) {
      // no policy target
      return false;
    } else {
      // dense case
      policy = dense_policy_start_ptr()[index.start];
      return true;
    }
  } else {
    throw util::Exception("Invalid policy tensor index (%d, %d) at state index %d", index.start,
                          index.end, state_index);
  }
}

template <concepts::Game Game>
bool GameLog<Game>::get_action_values(int state_index, ActionValueTensor& action_values) const {
  util::release_assert(state_index >= 0 && state_index < num_non_terminal_positions(),
                       "Invalid state index %d", state_index);

  tensor_index_t index = action_values_target_index_start_ptr()[state_index];

  if (index.start < index.end) {
    // sparse case
    action_values.setZero();
    const sparse_tensor_entry_t* sparse_start = sparse_action_values_entry_start_ptr();
    for (int i = index.start; i < index.end; ++i) {
      sparse_tensor_entry_t entry = sparse_start[i];
      action_values.data()[entry.offset] = entry.probability;
    }
    return true;
  } else if (index.start == index.end) {
    if (index.start < 0) {
      // no action_values target
      return false;
    } else {
      // dense case
      action_values = dense_action_values_start_ptr()[index.start];
      return true;
    }
  } else {
    throw util::Exception("Invalid action values tensor index (%d, %d) at state index %d",
                          index.start, index.end, state_index);
  }
}

template <concepts::Game Game>
const typename GameLog<Game>::State* GameLog<Game>::get_state(int state_index) const {
  return state_start_ptr() + state_index;
}

template <concepts::Game Game>
action_t GameLog<Game>::get_prev_action(int state_index) const {
  return state_index==0 ? -1 : action_start_ptr()[state_index-1];
}

template <concepts::Game Game>
typename GameLog<Game>::ValueTensor GameLog<Game>::get_outcome() const {
  return reinterpret_cast<ValueTensor*>(buffer_ + outcome_start_mem_offset())[0];
}

template <concepts::Game Game>
GameLogBase::pos_index_t GameLog<Game>::get_pos_index(int index) const {
  if (index < 0 || index >= num_sampled_positions()) {
    throw util::Exception("%s(%d) out of bounds in %s (%u)", __func__, index, filename_.c_str(),
                          num_sampled_positions());
  }

  return sampled_indices_start_ptr()[index];
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
  entry->terminal = false;
  entries_.push_back(entry);
  sample_count_ += use_for_training;
}

template <concepts::Game Game>
void GameLogWriter<Game>::add_terminal(const State& state, const ValueTensor& outcome) {
  if (terminal_added_) return;
  terminal_added_ = true;
  Entry* entry = new Entry();
  entry->position = state;
  entry->policy_target.setZero();
  entry->action_values.setZero();
  entry->action = -1;
  entry->use_for_training = false;
  entry->policy_target_is_valid = false;
  entry->action_values_are_valid = false;
  entry->terminal = true;
  entries_.push_back(entry);

  outcome_ = outcome;
}

template <concepts::Game Game>
void GameLogWriter<Game>::serialize(std::ostream& stream) const {
  using GameLog = core::GameLog<Game>;
  using Header = GameLog::Header;
  using pos_index_t = GameLog::pos_index_t;

  util::release_assert(!entries_.empty(), "Illegal serialization of empty GameLogWriter");
  int num_entries = entries_.size();
  int num_non_terminal_entries = num_entries - 1;

  std::vector<pos_index_t> sampled_indices;
  std::vector<action_t> actions;
  std::vector<seat_index_t> seat_indices;
  std::vector<tensor_index_t> policy_target_indices;
  std::vector<tensor_index_t> action_values_target_indices;
  std::vector<State> states;
  std::vector<PolicyTensor> dense_policy_tensors;
  std::vector<sparse_tensor_entry_t> sparse_policy_entries;
  std::vector<ActionValueTensor> dense_action_values_tensors;
  std::vector<sparse_tensor_entry_t> sparse_action_values_entries;

  sampled_indices.reserve(sample_count_);
  actions.reserve(num_non_terminal_entries);
  seat_indices.reserve(num_non_terminal_entries);
  policy_target_indices.reserve(num_non_terminal_entries);
  action_values_target_indices.reserve(num_non_terminal_entries);
  states.reserve(num_entries);
  dense_policy_tensors.reserve(num_non_terminal_entries);
  sparse_policy_entries.reserve(1 + num_non_terminal_entries * sizeof(PolicyTensor) /
                                        (2 * sizeof(sparse_tensor_entry_t)));
  dense_action_values_tensors.reserve(num_non_terminal_entries);
  sparse_action_values_entries.reserve(1 + num_non_terminal_entries * sizeof(ActionValueTensor) /
                                               (2 * sizeof(sparse_tensor_entry_t)));

  for (int move_num = 0; move_num < num_entries; ++move_num) {
    const Entry* entry = entries_[move_num];
    states.push_back(entry->position);

    if (entry->terminal) continue;

    if (entry->use_for_training) {
      sampled_indices.push_back(move_num);
    }

    actions.push_back(entry->action);
    seat_indices.push_back(entry->active_seat);

    tensor_index_t policy_target_index = {-1, -1};
    if (entry->policy_target_is_valid) {
      policy_target_index =
          write_target(entry->policy_target, dense_policy_tensors, sparse_policy_entries);
    }
    policy_target_indices.push_back(policy_target_index);

    tensor_index_t action_values_target_index = {-1, -1};
    if (entry->action_values_are_valid) {
      action_values_target_index = write_target(
          entry->action_values, dense_action_values_tensors, sparse_action_values_entries);
    }
    action_values_target_indices.push_back(action_values_target_index);
  }

  Header header;
  header.num_samples = sample_count_;
  header.num_positions = num_entries;
  header.num_dense_policies = dense_policy_tensors.size();
  header.num_sparse_policy_entries = sparse_policy_entries.size();
  header.num_dense_action_values = dense_action_values_tensors.size();
  header.num_sparse_action_values_entries = sparse_action_values_entries.size();
  header.extra = 0;

  write_section(stream, &header);
  write_section(stream, &outcome_);
  write_section(stream, sampled_indices.data(), sampled_indices.size());
  write_section(stream, actions.data(), actions.size());
  write_section(stream, seat_indices.data(), seat_indices.size());
  write_section(stream, policy_target_indices.data(), policy_target_indices.size());
  write_section(stream, action_values_target_indices.data(), action_values_target_indices.size());
  write_section(stream, states.data(), states.size());
  write_section(stream, dense_policy_tensors.data(), dense_policy_tensors.size());
  write_section(stream, sparse_policy_entries.data(), sparse_policy_entries.size());
  write_section(stream, dense_action_values_tensors.data(), dense_action_values_tensors.size());
  write_section(stream, sparse_action_values_entries.data(), sparse_action_values_entries.size());
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
void GameLogWriter<Game>::write_section(std::ostream& os, const T* t, int count) {
  constexpr int A = GameLogBase::kAlignment;
  int n_bytes = sizeof(T) * count;
  os.write(reinterpret_cast<const char*>(t), n_bytes);

  int remainder = n_bytes % A;
  if (remainder) {
    int padding = A - remainder;
    uint8_t zeroes[A] = {0};
    os.write(reinterpret_cast<const char*>(zeroes), padding);
  }
}

template <concepts::Game Game>
template <eigen_util::concepts::FTensor Tensor>
GameLogBase::tensor_index_t GameLogWriter<Game>::write_target(
    const Tensor& target, std::vector<Tensor>& dense_tensors,
    std::vector<sparse_tensor_entry_t>& sparse_tensor_entries) {
  int num_nonzero_entries = eigen_util::count(target);

  if (num_nonzero_entries == 0) {
    return {-1, -1};
  }

  int sparse_repr_size = sizeof(sparse_tensor_entry_t) * num_nonzero_entries;
  int dense_repr_size = sizeof(Tensor);

  if (sparse_repr_size * 2 > dense_repr_size) {
    // use dense representation
    int index = dense_tensors.size();
    if (index > std::numeric_limits<int16_t>::max()) {
      throw util::Exception("Too many dense tensor entries (%d)", index);
    }
    int16_t index16 = index;
    dense_tensors.push_back(target);
    return {index16, index16};
  }

  int start = sparse_tensor_entries.size();

  constexpr int N = Tensor::Dimensions::total_size;
  const auto* data = target.data();
  for (int i = 0; i < N; ++i) {
    if (data[i]) {
      sparse_tensor_entries.emplace_back(i, data[i]);
    }
  }

  int end = sparse_tensor_entries.size();

  if (end > std::numeric_limits<int16_t>::max()) {
    throw util::Exception("Too many sparse tensor entries (%d)", end);
  }
  int16_t start16 = start;
  int16_t end16 = end;
  return {start16, end16};
}

}  // namespace core
