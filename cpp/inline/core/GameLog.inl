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
GameLog<Game>::MemOffsetTable::MemOffsetTable(const Header& header) {
  int ns = header.num_samples;
  int np = header.num_positions;
  int nntp = np - 1;  // num non-terminal positions
  int nse = header.num_sparse_entries;

  outcome = align(sizeof(Header));
  sampled_indices = outcome + align(sizeof(ValueTensor));
  action_types = sampled_indices + align(ns * sizeof(pos_index_t));

  if (kSingleActionTypeOptimization && kNumActionTypes == 1) {
    actions = action_types;
  } else {
    actions = action_types + align(nntp * sizeof(action_type_t));
  }

  policy_target_indices = actions + align(nntp * sizeof(action_t));
  action_values_target_indices = policy_target_indices + align(nntp * sizeof(tensor_index_t));
  states = action_values_target_indices + align(nntp * sizeof(tensor_index_t));
  sparse_tensor_entries = states + align(np * sizeof(State));
  dense_tensors[0] = sparse_tensor_entries + align(nse * sizeof(sparse_tensor_entry_t));

  for (action_type_t a = 1; a < kNumActionTypes; ++a) {
    ActionTypeDispatcher::call(a, [&](auto A) {
      using Tensor = mp::TypeAt_t<PolicyTensorVariant, A>;
      int n = header.num_dense_policies_per_action_type[a - 1];
      dense_tensors[a] = dense_tensors[a - 1] + align(n * sizeof(Tensor));
    });
  }
}

template <concepts::Game Game>
GameLog<Game>::GameLog(const char* filename)
    : filename_(filename),
      buffer_(get_buffer()),
      mem_offsets_(header()) {
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
    info_array[1 + a].template init<Tensor>(Target::name(), a);
  });

  return info_array;
}

template <concepts::Game Game>
void GameLog<Game>::load(int index, bool apply_symmetry, float* input_values, int* target_indices,
                         float** target_arrays) const {
  util::release_assert(index >= 0 && index < num_sampled_positions(),
                       "Index %d out of bounds [0, %d) in %s", index, num_sampled_positions(),
                       filename_.c_str());

  pos_index_t state_index = get_pos_index(index);
  PolicyTensorVariant policy = get_policy(state_index);
  PolicyTensorVariant next_policy = get_policy(state_index + 1);
  ActionValueTensorVariant action_values = get_action_values(state_index);

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

  for (int i = 0; i < num_states_to_cp; ++i) {
    Game::Symmetries::apply(states[i], sym);
  }
  Game::Symmetries::apply(final_state, sym);
  Game::Symmetries::apply(policy, sym);
  Game::Symmetries::apply(next_policy, sym);
  Game::Symmetries::apply(action_values, sym);

  ValueTensor outcome = get_outcome();

  auto input = InputTensorizor::tensorize(start_pos, cur_pos);
  memcpy(input_values, input.data(), input.size() * sizeof(float));

  GameLogView view{cur_pos, &final_state, &outcome, &policy, &next_policy, &action_values};

  constexpr size_t N = mp::Length_v<TrainingTargetsList>;
  using TrainingTargetsDispatcher = util::IndexedDispatcher<N>;

  for (int t = 0;; ++t) {
    int target_index = target_indices[t];
    if (target_index < 0) break;

    TrainingTargetsDispatcher::call(target_index, [&](auto a) {
      if (target_index == a) {
        using Target = mp::TypeAt_t<TrainingTargetsList, a>;
        auto tensor = Target::tensorize(view);
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
    action_t last_action = get_prev_action(i);
    Game::IO::print_state(std::cout, *pos, last_action);

    // TODO: print policy if available
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
typename GameLog<Game>::Header& GameLog<Game>::header() {
  return *reinterpret_cast<Header*>(buffer_);
}

template <concepts::Game Game>
const typename GameLog<Game>::Header& GameLog<Game>::header() const {
  return *reinterpret_cast<const Header*>(buffer_);
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
const action_t* GameLog<Game>::action_type_start_ptr() const {
  return reinterpret_cast<action_t*>(buffer_ + mem_offsets_.action_types);
}

template <concepts::Game Game>
const action_t* GameLog<Game>::action_start_ptr() const {
  return reinterpret_cast<action_t*>(buffer_ + mem_offsets_.actions);
}

template <concepts::Game Game>
const GameLogBase::tensor_index_t* GameLog<Game>::policy_target_index_start_ptr() const {
  return reinterpret_cast<tensor_index_t*>(buffer_ + mem_offsets_.policy_target_indices);
}

template <concepts::Game Game>
const GameLogBase::tensor_index_t* GameLog<Game>::action_values_target_index_start_ptr() const {
  return reinterpret_cast<tensor_index_t*>(buffer_ + mem_offsets_.action_values_target_indices);
}

template <concepts::Game Game>
const typename GameLog<Game>::State* GameLog<Game>::state_start_ptr() const {
  return reinterpret_cast<State*>(buffer_ + mem_offsets_.states);
}

template <concepts::Game Game>
const GameLogBase::sparse_tensor_entry_t* GameLog<Game>::sparse_tensor_entry_start_ptr() const {
  return reinterpret_cast<sparse_tensor_entry_t*>(buffer_ + mem_offsets_.sparse_tensor_entries);
}

template <concepts::Game Game>
template <action_type_t ActionType>
const auto* GameLog<Game>::dense_tensor_start_ptr() const {
  using Tensor = mp::TypeAt_t<PolicyTensorVariant, ActionType>;
  return reinterpret_cast<Tensor*>(buffer_ + mem_offsets_.dense_tensors[ActionType]);
}

template <concepts::Game Game>
const GameLogBase::pos_index_t* GameLog<Game>::sampled_indices_start_ptr() const {
  return reinterpret_cast<pos_index_t*>(buffer_ + mem_offsets_.sampled_indices);
}

template <concepts::Game Game>
typename GameLog<Game>::PolicyTensorVariant GameLog<Game>::get_policy(int state_index) const {
  action_type_t type = get_action_type(state_index);

  util::release_assert(type >= 0 && type < kNumActionTypes,
                       "Invalid action type %d at index %d in game log %s", type, state_index,
                       filename_.c_str());

  return ActionTypeDispatcher::call(type, [&](auto A) {
    PolicyTensorVariant policy(std::in_place_index<A>);
    auto& policy_tensor = std::get<A>(policy);
    if (state_index >= num_non_terminal_positions()) {
      policy_tensor.setZero();
      return policy;
    }

    tensor_index_t index = policy_target_index_start_ptr()[state_index];

    if (index.start < index.end) {
      // sparse case
      policy_tensor.setZero();
      const sparse_tensor_entry_t* sparse_start = sparse_tensor_entry_start_ptr();
      for (int i = index.start; i < index.end; ++i) {
        sparse_tensor_entry_t entry = sparse_start[i];
        policy_tensor(entry.offset) = entry.probability;
      }
      return policy;
    } else if (index.start == index.end) {
      if (index.start < 0) {
        // no policy target
        policy_tensor.setZero();
        return policy;
      } else {
        // dense case
        return dense_tensor_start_ptr<A>()[index.start];
      }
    } else {
      throw util::Exception("Invalid policy tensor index (%d, %d) at state index %d", index.start,
                            index.end, state_index);
    }
  });
}

template <concepts::Game Game>
typename GameLog<Game>::ActionValueTensorVariant
GameLog<Game>::get_action_values(int state_index) const {
  action_type_t type = get_action_type(state_index);

  util::release_assert(type >= 0 && type < kNumActionTypes,
                       "Invalid action type %d at index %d in game log %s", type, state_index,
                       filename_.c_str());

  return ActionTypeDispatcher::call(type, [&](auto A) {
    ActionValueTensorVariant action_values(std::in_place_index<A>);
    auto& action_values_tensor = std::get<A>(action_values);
    if (state_index >= num_non_terminal_positions()) {
      action_values_tensor.setZero();
      return action_values;
    }

    tensor_index_t index = action_values_target_index_start_ptr()[state_index];

    if (index.start < index.end) {
      // sparse case
      action_values_tensor.setZero();
      const sparse_tensor_entry_t* sparse_start = sparse_tensor_entry_start_ptr();
      for (int i = index.start; i < index.end; ++i) {
        sparse_tensor_entry_t entry = sparse_start[i];
        action_values_tensor(entry.offset) = entry.probability;
      }
      return action_values;
    } else if (index.start == index.end) {
      if (index.start < 0) {
        // no action_values target
        action_values_tensor.setZero();
        return action_values;
      } else {
        // dense case
        return dense_tensor_start_ptr<A>()[index.start];
      }
    } else {
      throw util::Exception("Invalid action values tensor index (%d, %d) at state index %d",
                            index.start, index.end, state_index);
    }
  });
}

template <concepts::Game Game>
action_type_t GameLog<Game>::get_action_type(int state_index) const {
  if (kNumActionTypes == 1) return 0;
  return action_type_start_ptr() + state_index;
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
  return reinterpret_cast<ValueTensor*>(buffer_ + mem_offsets_.outcome)[0];
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
void GameLogWriter<Game>::add(const State& state, action_type_t action_type, action_t action,
                              const PolicyTensorVariant* policy_target,
                              const ActionValueTensorVariant* action_values,
                              bool use_for_training) {
  // TODO: get entries from a thread-specific object pool
  Entry* entry = new Entry();
  entry->position = state;
  ActionTypeDispatcher::call(action_type, [&](auto A) {
    if (policy_target) {
      entry->policy_target = *policy_target;
    } else {
      entry->policy_target = PolicyTensorVariant(std::in_place_index<A>);
      std::get<A>(entry->policy_target).setZero();
    }
    if (action_values) {
      entry->action_values = *action_values;
    } else {
      entry->action_values = ActionValueTensorVariant(std::in_place_index<A>);
      std::get<A>(entry->action_values).setZero();
    }
  });
  entry->action_type = action_type;
  entry->action = action;
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

  entry->action_type = 0;
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
  std::vector<action_type_t> action_types;
  std::vector<action_t> actions;
  std::vector<tensor_index_t> policy_target_indices;
  std::vector<tensor_index_t> action_values_target_indices;
  std::vector<State> states;
  std::vector<sparse_tensor_entry_t> sparse_tensor_entries;
  tensor_vector_tuple_t dense_tensors;

  sampled_indices.reserve(sample_count_);
  action_types.reserve(num_non_terminal_entries);
  actions.reserve(num_non_terminal_entries);
  policy_target_indices.reserve(num_non_terminal_entries);
  action_values_target_indices.reserve(num_non_terminal_entries);
  states.reserve(num_entries);

  // TODO: reserve space for dense and sparse tensors as optimization

  for (int move_num = 0; move_num < num_entries; ++move_num) {
    const Entry* entry = entries_[move_num];
    states.push_back(entry->position);

    if (entry->terminal) continue;

    if (entry->use_for_training) {
      sampled_indices.push_back(move_num);
    }

    action_types.push_back(entry->action_type);
    actions.push_back(entry->action);

    tensor_index_t policy_target_index = {-1, -1};
    if (entry->policy_target_is_valid) {
      policy_target_index = write_target(entry->action_type, entry->policy_target, dense_tensors,
                                         sparse_tensor_entries);
    }
    policy_target_indices.push_back(policy_target_index);

    tensor_index_t action_values_target_index = {-1, -1};
    if (entry->action_values_are_valid) {
      action_values_target_index = write_target(entry->action_type, entry->action_values,
                                                dense_tensors, sparse_tensor_entries);
    }
    action_values_target_indices.push_back(action_values_target_index);
  }

  Header header;
  header.num_samples = sample_count_;
  header.num_positions = num_entries;
  header.num_sparse_entries = sparse_tensor_entries.size();

  for (action_type_t a = 0; a < kNumActionTypes; ++a) {
    ActionTypeDispatcher::call(a, [&](auto A) {
      header.num_dense_policies_per_action_type[a] = std::get<A>(dense_tensors).size();
    });
  }

  write_section(stream, &header);
  write_section(stream, &outcome_);
  write_section(stream, sampled_indices.data(), sampled_indices.size());

  if (!GameLogBase::kSingleActionTypeOptimization || kNumActionTypes > 1) {
    write_section(stream, action_types.data(), action_types.size());
  }

  write_section(stream, actions.data(), actions.size());
  write_section(stream, policy_target_indices.data(), policy_target_indices.size());
  write_section(stream, action_values_target_indices.data(), action_values_target_indices.size());
  write_section(stream, states.data(), states.size());
  write_section(stream, sparse_tensor_entries.data(), sparse_tensor_entries.size());

  for (action_type_t a = 0; a < kNumActionTypes; ++a) {
    ActionTypeDispatcher::call(a, [&](auto A) {
      const auto& vec = std::get<A>(dense_tensors);
      write_section(stream, vec.data(), vec.size());
    });
  }
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
GameLogBase::tensor_index_t GameLogWriter<Game>::write_target(
    action_type_t action_type, const PolicyTensorVariant& target,
    tensor_vector_tuple_t& dense_tensors,
    std::vector<sparse_tensor_entry_t>& sparse_tensor_entries) {
  return ActionTypeDispatcher::call(action_type, [&](auto A) {
    using Tensor = mp::TypeAt_t<PolicyTensorVariant, A>;
    const Tensor& target_tensor = std::get<A>(target);
    std::vector<Tensor>& dense_tensors_vec = std::get<A>(dense_tensors);

    int num_nonzero_entries = eigen_util::count(std::get<A>(target_tensor));

    if (num_nonzero_entries == 0) {
      return tensor_index_t{-1, -1};
    }

    int sparse_repr_size = sizeof(sparse_tensor_entry_t) * num_nonzero_entries;
    int dense_repr_size = sizeof(Tensor);

    if (sparse_repr_size * 2 > dense_repr_size) {
      // use dense representation
      int index = dense_tensors_vec.size();
      if (index > std::numeric_limits<int16_t>::max()) {
        throw util::Exception("Too many sparse tensor entries (%d)", index);
      }
      int16_t index16 = index;
      dense_tensors_vec.push_back(target_tensor);
      return tensor_index_t{index16, index16};
    }

    int start = sparse_tensor_entries.size();

    constexpr int N = Tensor::Dimensions::total_size;
    const auto* data = target_tensor.data();
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
    return tensor_index_t{start16, end16};
  });
}

}  // namespace core
