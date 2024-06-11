#include <core/GameLog.hpp>

#include <util/BitSet.hpp>
#include <util/EigenUtil.hpp>
#include <util/LoggingUtil.hpp>
#include <util/Math.hpp>

#include <limits>

namespace core {

template <eigen_util::concepts::FTensor Tensor>
void ShapeInfo::init(const char* name, int target_index) {
  using Shape = eigen_util::extract_shape_t<Tensor>;
  this->name = name;
  this->dims = new int[Shape::count];
  this->num_dims = Shape::count;
  this->target_index = target_index;

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
      non_sym_sample_index_start_mem_offset_(non_sym_sample_index_start_mem_offset()),
      action_start_mem_offset_(action_start_mem_offset()),
      policy_tensor_index_start_mem_offset_(policy_tensor_index_start_mem_offset()),
      snapshot_start_mem_offset_(snapshot_start_mem_offset()),
      dense_policy_start_mem_offset_(dense_policy_start_mem_offset()),
      sparse_policy_entry_start_mem_offset_(sparse_policy_entry_start_mem_offset()) {
  util::release_assert(num_positions() > 0, "Empty game log file: %s", filename_.c_str());
}

template <concepts::Game Game>
GameLog<Game>::~GameLog() {
  delete[] buffer_;
}

template <concepts::Game Game>
ShapeInfo* GameLog<Game>::get_shape_info_array() {
  using TargetList = typename TrainingTargetTensorizor::TargetList;
  constexpr int n_targets = mp::Length_v<TargetList>;
  constexpr int n = n_targets + 2;  // 1 for input, 1 for terminator

  ShapeInfo* info_array = new ShapeInfo[n];
  info_array[0].template init<InputTensor>("input", -1);

  mp::constexpr_for<0, n_targets, 1>([&](auto a) {
    using Target = mp::TypeAt_t<TargetList, a>;
    using Tensor = typename Target::Tensor;
    info_array[1 + a].template init<Tensor>(Target::kName, a);
  });

  return info_array;
}

template <concepts::Game Game>
void GameLog<Game>::load(int index, bool apply_symmetry, float** input_values, int* target_indices,
                         float** target_value_arrays) {
  int state_index;
  symmetry_index_t sym_index = -1;

  if (apply_symmetry) {
    sym_sample_index_t sym_sample_index = get_sym_sample_index(index);
    state_index = sym_sample_index.state_index;
    sym_index = sym_sample_index.sym_index;
  } else {
    non_sym_sample_index_t non_sym_sample_index = get_non_sym_sample_index(index);
    state_index = non_sym_sample_index.state_index;
  }

  util::release_assert(state_index >= 0 && state_index < num_positions(),
                       "Invalid state index %d for index=%d in %s", state_index, index,
                       filename_.c_str());

  PolicyTensor policy = get_policy(state_index);
  PolicyTensor next_policy = get_policy(state_index + 1);
  StateSnapshot* cur_pos = get_snapshot(state_index);
  StateSnapshot* start_pos = get_snapshot(std::max(0, state_index - Game::kHistorySize));
  StateSnapshot* final_pos = get_snapshot(num_positions() - 1);

  if (sym_index >= 0) {
    Transform* transform = core::Transforms<Game>::get(sym_index);
    for (StateSnapshot* pos = start_pos; pos <= cur_pos; ++pos) {
      transform->apply(*pos);
    }
    if (final_pos > cur_pos) {
      transform->apply(*final_pos);
    }
    transform->apply(policy);
    transform->apply(next_policy);
  }

  seat_index_t cp = Rules::current_player(*cur_pos);
  ValueArray outcome = get_outcome();
  eigen_util::left_rotate(outcome, cp);

  auto input = InputTensorizor::tensorize(start_pos, cur_pos);
  memcpy(*input_values, input.data(), input.size() * sizeof(float));

  core::GameLogView<Game> view{cur_pos, final_pos, &outcome, &policy, &next_policy};

  using TargetList = typename TrainingTargetTensorizor::TargetList;
  constexpr size_t N = mp::Length_v<TargetList>;

  int t = 0;
  while (true) {
    int target_index = target_indices[t++];
    if (target_index < 0) break;

    mp::constexpr_for<0, N, 1>([&](auto a) {
      if (target_index == a) {
        using Target = mp::TypeAt_t<TargetList, a>;
        auto tensor = Target::tensorize(view);
        memcpy(target_value_arrays[t], tensor.data(), tensor.size() * sizeof(float));
        ++t;
      }
    });
  }
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
int GameLog<Game>::num_samples_with_symmetry_expansion() const {
  return header().num_samples_with_symmetry_expansion;
}

template <concepts::Game Game>
int GameLog<Game>::num_samples_without_symmetry_expansion() const {
  return header().num_samples_without_symmetry_expansion;
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
constexpr int GameLog<Game>::header_start_mem_offset() {
  return 0;
}

template <concepts::Game Game>
constexpr int GameLog<Game>::outcome_start_mem_offset() {
  return header_start_mem_offset() + align(sizeof(Header));
}

template <concepts::Game Game>
constexpr int GameLog<Game>::sym_sample_index_start_mem_offset() {
  return outcome_start_mem_offset() + align(sizeof(ValueArray));
}

template <concepts::Game Game>
int GameLog<Game>::non_sym_sample_index_start_mem_offset() const {
  return sym_sample_index_start_mem_offset() +
         align(num_samples_with_symmetry_expansion() * sizeof(sym_sample_index_t));
}

template <concepts::Game Game>
int GameLog<Game>::action_start_mem_offset() const {
  return non_sym_sample_index_start_mem_offset_ +
         align(num_samples_without_symmetry_expansion() * sizeof(non_sym_sample_index_t));
}

template <concepts::Game Game>
int GameLog<Game>::policy_tensor_index_start_mem_offset() const {
  return action_start_mem_offset_ + align(num_non_terminal_positions() * sizeof(action_t));
}

template <concepts::Game Game>
int GameLog<Game>::snapshot_start_mem_offset() const {
  return policy_tensor_index_start_mem_offset_ +
         align(num_non_terminal_positions() * sizeof(policy_tensor_index_t));
}

template <concepts::Game Game>
int GameLog<Game>::dense_policy_start_mem_offset() const {
  return snapshot_start_mem_offset_ + align(num_non_terminal_positions() * sizeof(StateSnapshot));
}

template <concepts::Game Game>
int GameLog<Game>::sparse_policy_entry_start_mem_offset() const {
  return dense_policy_start_mem_offset_ + align(num_dense_policies() * sizeof(PolicyTensor));
}

template <concepts::Game Game>
GameLogBase::policy_tensor_index_t* GameLog<Game>::policy_tensor_index_start_ptr() {
  return reinterpret_cast<policy_tensor_index_t*>(buffer_ + policy_tensor_index_start_mem_offset_);
}

template <concepts::Game Game>
typename GameLog<Game>::StateSnapshot* GameLog<Game>::snapshot_start_ptr() {
  return reinterpret_cast<StateSnapshot*>(buffer_ + snapshot_start_mem_offset_);
}

template <concepts::Game Game>
typename GameLog<Game>::PolicyTensor* GameLog<Game>::dense_policy_start_ptr() {
  return reinterpret_cast<PolicyTensor*>(buffer_ + dense_policy_start_mem_offset_);
}

template <concepts::Game Game>
GameLogBase::sparse_policy_entry_t* GameLog<Game>::sparse_policy_entry_start_ptr() {
  return reinterpret_cast<sparse_policy_entry_t*>(buffer_ + sparse_policy_entry_start_mem_offset_);
}

template <concepts::Game Game>
GameLogBase::sym_sample_index_t* GameLog<Game>::sym_sample_index_start_ptr() {
  return reinterpret_cast<sym_sample_index_t*>(buffer_ + sym_sample_index_start_mem_offset());
}

template <concepts::Game Game>
GameLogBase::non_sym_sample_index_t* GameLog<Game>::non_sym_sample_index_start_ptr() {
  return reinterpret_cast<non_sym_sample_index_t*>(buffer_ +
                                                   non_sym_sample_index_start_mem_offset_);
}

template <concepts::Game Game>
typename GameLog<Game>::PolicyTensor GameLog<Game>::get_policy(int state_index) {
  PolicyTensor policy;
  if (state_index >= num_non_terminal_positions()) {
    policy.setZero();
    return policy;
  }

  policy_tensor_index_t index = policy_tensor_index_start_ptr()[state_index];

  if (index.start < index.end) {
    // sparse case
    sparse_policy_entry_t* sparse_start = sparse_policy_entry_start_ptr();
    for (int i = index.start; i < index.end; ++i) {
      sparse_policy_entry_t entry = sparse_start[i];
      policy(entry.offset) = entry.probability;
    }
    return policy;
  } else if (index.start == index.end) {
    if (index.start < 0) {
      // no policy target
      policy.setZero();
      return policy;
    } else {
      // dense case
      return dense_policy_start_ptr()[index.start];
    }
  } else {
    throw util::Exception("Invalid policy tensor index (%d, %d) at state index %d", index.start,
                          index.end, state_index);
  }
}

template <concepts::Game Game>
typename GameLog<Game>::StateSnapshot* GameLog<Game>::get_snapshot(int state_index) {
  return snapshot_start_ptr() + state_index;
}

template <concepts::Game Game>
GameLogBase::sym_sample_index_t GameLog<Game>::get_sym_sample_index(int index) {
  if (index < 0 || index >= num_samples_with_symmetry_expansion()) {
    throw util::Exception("%s(%d) out of bounds in %s (%u)", __func__, index,
                          filename_.c_str(), num_samples_with_symmetry_expansion());
  }

  return sym_sample_index_start_ptr()[index];
}

template <concepts::Game Game>
GameLogBase::non_sym_sample_index_t GameLog<Game>::get_non_sym_sample_index(int index) {
  if (index < 0 || index >= num_samples_without_symmetry_expansion()) {
    throw util::Exception("%s(%d) out of bounds in %s (%u)", __func__, index,
                          filename_.c_str(), num_samples_without_symmetry_expansion());
  }

  return non_sym_sample_index_start_ptr()[index];
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
void GameLogWriter<Game>::add(const FullState& state, action_index_t action,
                              const PolicyTensor* policy_target, bool use_for_training) {
  // TODO: get entries from a thread-specific object pool
  Entry* entry = new Entry();
  entry->position = state.current();
  entry->symmetries = Rules::get_symmetry_indices(state);
  if (policy_target) {
    entry->policy_target = *policy_target;
  } else {
    entry->policy_target.setZero();
  }
  entry->action = action;
  entry->use_for_training = use_for_training;
  entry->policy_target_is_valid = policy_target != nullptr;
  entry->terminal = false;
  entries_.push_back(entry);
  if (use_for_training) {
    sym_train_count_ += entry->symmetries.count();
    non_sym_train_count_++;
  }
}

template <concepts::Game Game>
void GameLogWriter<Game>::add_terminal(const FullState& state, const ValueArray& outcome) {
  if (terminal_added_) return;
  terminal_added_ = true;
  Entry* entry = new Entry();
  entry->position = state.current();
  entry->policy_target.setZero();
  entry->action = -1;
  entry->use_for_training = false;
  entry->policy_target_is_valid = false;
  entry->terminal = true;
  entries_.push_back(entry);

  outcome_ = outcome;
}

template <concepts::Game Game>
void GameLogWriter<Game>::serialize(std::ostream& stream) const {
  using GameLog = core::GameLog<Game>;
  using Header = typename GameLog::Header;
  using sym_sample_index_t = typename GameLog::sym_sample_index_t;
  using non_sym_sample_index_t = typename GameLog::non_sym_sample_index_t;
  using policy_tensor_index_t = typename GameLog::policy_tensor_index_t;
  using sparse_policy_entry_t = typename GameLog::sparse_policy_entry_t;

  util::release_assert(!entries_.empty(), "Illegal serialization of empty GameLogWriter");
  int num_entries = entries_.size();
  int num_non_terminal_entries = num_entries - 1;

  std::vector<sym_sample_index_t> sym_sample_indices;
  std::vector<non_sym_sample_index_t> non_sym_sample_indices;
  std::vector<action_t> actions;
  std::vector<policy_tensor_index_t> policy_tensor_indices;
  std::vector<StateSnapshot> snapshots;
  std::vector<PolicyTensor> dense_policy_tensors;
  std::vector<sparse_policy_entry_t> sparse_policy_entries;

  sym_sample_indices.reserve(sym_train_count_);
  non_sym_sample_indices.reserve(non_sym_train_count_);
  actions.reserve(num_non_terminal_entries);
  policy_tensor_indices.reserve(num_non_terminal_entries);
  snapshots.reserve(num_entries);
  dense_policy_tensors.reserve(num_non_terminal_entries);
  sparse_policy_entries.reserve(1 + num_non_terminal_entries * sizeof(PolicyTensor) /
                                        (2 * sizeof(sparse_policy_entry_t)));

  for (int move_num = 0; move_num < num_entries; ++move_num) {
    const Entry* entry = entries_[move_num];
    snapshots.push_back(entry->position);

    if (entry->terminal) continue;

    if (entry->use_for_training) {
      for (symmetry_index_t sym_index : bitset_util::on_indices(entry->symmetries)) {
        sym_sample_indices.emplace_back(move_num, sym_index);
      }
      non_sym_sample_indices.emplace_back(move_num);
    }

    actions.push_back(entry->action);
    policy_tensor_indices.push_back(
        write_policy_target(*entry, dense_policy_tensors, sparse_policy_entries));
  }

  Header header;
  header.num_samples_with_symmetry_expansion = sym_train_count_;
  header.num_samples_without_symmetry_expansion = non_sym_train_count_;
  header.num_positions = num_entries;
  header.num_dense_policies = dense_policy_tensors.size();
  header.num_sparse_policy_entries = sparse_policy_entries.size();
  header.extra = 0;

  write_section(stream, &header);
  write_section(stream, &outcome_);
  write_section(stream, sym_sample_indices.data(), sym_sample_indices.size());
  write_section(stream, non_sym_sample_indices.data(), non_sym_sample_indices.size());
  write_section(stream, actions.data(), actions.size());
  write_section(stream, policy_tensor_indices.data(), policy_tensor_indices.size());
  write_section(stream, snapshots.data(), snapshots.size());
  write_section(stream, dense_policy_tensors.data(), dense_policy_tensors.size());
  write_section(stream, sparse_policy_entries.data(), sparse_policy_entries.size());
}

template <concepts::Game Game>
bool GameLogWriter<Game>::is_previous_entry_used_for_training() const {
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
GameLogBase::policy_tensor_index_t GameLogWriter<Game>::write_policy_target(
    const Entry& entry, std::vector<PolicyTensor>& dense_tensors,
    std::vector<sparse_policy_entry_t>& sparse_tensor_entries) {
  if (!entry.policy_target_is_valid) {
    return {-1, -1};
  }

  const PolicyTensor& policy_target = entry.policy_target;
  int num_nonzero_entries = eigen_util::count(policy_target);

  int sparse_repr_size = sizeof(sparse_policy_entry_t) * num_nonzero_entries;
  int dense_repr_size = sizeof(PolicyTensor);

  if (sparse_repr_size * 2 > dense_repr_size) {
    // use dense representation
    int index = dense_tensors.size();
    if (index > std::numeric_limits<int16_t>::max()) {
      throw util::Exception("Too many sparse tensor entries (%d)", index);
    }
    int16_t index16 = index;
    dense_tensors.push_back(policy_target);
    return {index16, index16};
  }

  int start = sparse_tensor_entries.size();

  constexpr int N = eigen_util::extract_shape_t<PolicyTensor>::total_size;
  const auto* data = policy_target.data();
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
