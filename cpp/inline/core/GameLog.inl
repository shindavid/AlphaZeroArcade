#include <core/GameLog.hpp>

#include <util/BitSet.hpp>
#include <util/EigenUtil.hpp>
#include <util/LoggingUtil.hpp>

#include <limits>

namespace core {

template <eigen_util::concepts::FTensor Tensor>
void ShapeInfo::init(const char* name) {
  using Shape = eigen_util::extract_shape_t<Tensor>;
  this->name = name;
  this->dims = new int[Shape::count];
  this->num_dims = Shape::count;

  Shape shape;
  for (int i = 0; i < Shape::count; ++i) {
    this->dims[i] = shape[i];
  }
}

ShapeInfo::~ShapeInfo() {
  delete[] dims;
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
  entry->position = state.cur_pos();
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
void GameLogWriter<Game>::add_terminal(const FullState& state, const GameOutcome& outcome) {
  if (terminal_added_) return;
  terminal_added_ = true;
  Entry* entry = new Entry();
  entry->position = state.cur_pos();
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

  Header header;
  header.num_samples_with_symmetry_expansion = sym_train_count_;
  header.num_samples_without_symmetry_expansion = non_sym_train_count_;
  header.num_positions = num_entries;
  header.extra = 0;

  std::ostringstream state_data_stream;
  std::ostringstream aux_data_stream;

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
void GameLogWriter<Game>::write_section(std::ostream& os, const T* t, int count = 1) {
  int n_bytes = sizeof(T) * count;
  os.write(reinterpret_cast<const char*>(t), n_bytes);

  int remainder = n_bytes % 8;
  if (remainder) {
    int padding = 8 - remainder;
    uint8_t zeroes[8] = {0};
    os.write(reinterpret_cast<const char*>(zeroes), padding);
  }
}

template <concepts::Game Game>
GameLogBase::policy_tensor_index_t GameLogWriter<Game>::write_policy_target(
    const Entry& entry, std::vector<PolicyTensor>& dense_tensors,
    std::vector<GameLogBase::sparse_policy_entry_t>& sparse_tensor_entries) {
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
    dense_tensors.push_back(policy_target);
    return {index, index};
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
  return {start, end};
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
GameReadLog<GameState, Tensorizor>::GameReadLog(const char* filename) {
  filename_ = filename;
  file_ = fopen(filename, "rb");
  util::release_assert(file_, "Failed to open file '%s' for reading", filename);

  if (fread(&header_, sizeof(Header), 1, file_) != 1) {
    fclose(file_);
    file_ = nullptr;
    throw util::Exception("Failed to read header from file '%s'", filename);
  }

  if (fread(&outcome_, sizeof(GameOutcome), 1, file_) != 1) {
    fclose(file_);
    file_ = nullptr;
    throw util::Exception("Failed to read outcome from file '%s'", filename);
  }
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
GameReadLog<GameState, Tensorizor>::~GameReadLog() {
  if (file_) fclose(file_);
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
void GameReadLog<GameState, Tensorizor>::load(int index, bool apply_symmetry,
                                              const char** keys, float** values, int num_keys) {
  util::release_assert(file_, "Attempt to read from closed GameReadLog");

  int state_index;
  core::symmetry_index_t sym_index = -1;

  if (apply_symmetry) {
    if (index < 0 || index >= (int)header_.num_samples_with_symmetry_expansion) {
      throw util::Exception("Index %d(%d) out of bounds in GameReadLog (%u)", index, apply_symmetry,
                            header_.num_samples_with_symmetry_expansion);
    }

    SymSampleIndex sym_sample_index;
    seek_and_read(symmetric_index_offset(index), &sym_sample_index);

    state_index = sym_sample_index.state_index;
    sym_index = sym_sample_index.symmetry_index;
  } else {
    if (index < 0 || index >= (int)header_.num_samples_without_symmetry_expansion) {
      throw util::Exception("Index %d(%d) out of bounds in GameReadLog (%u)", index, apply_symmetry,
                            header_.num_samples_without_symmetry_expansion);
    }

    NonSymSampleIndex non_sym_sample_index;
    seek_and_read(non_symmetric_index_offset(index), &non_sym_sample_index);

    state_index = non_sym_sample_index;
  }

  AuxMemOffset aux_mem_offset;
  seek_and_read(aux_index_offset(state_index), &aux_mem_offset);

  AuxMemOffset next_aux_mem_offset = -1;
  if (state_index + 1 < (int)header_.num_game_states) {
    seek_and_read(aux_index_offset(state_index + 1), &next_aux_mem_offset);
  }

  GameStateHistory state_history;
  int history_start_index = std::max(0, state_index - Tensorizor::kHistorySize);
  for (int h = history_start_index; h < state_index; ++h) {
    GameStateData prev_state_data;
    seek_and_read(state_data_offset(h), &prev_state_data);
    state_history.push_back(prev_state_data);
  }

  GameStateData cur_state;
  seek_and_read(state_data_offset(state_index), &cur_state);

  GameStateData final_state;
  seek_and_read(state_data_offset(header_.num_game_states - 1), &final_state);

  PolicyTensor policy_tensor;
  load_policy(&policy_tensor, aux_mem_offset);

  PolicyTensor next_policy_tensor;
  load_policy(&next_policy_tensor, next_aux_mem_offset);

  if (sym_index >= 0) {
    Transform* transform = Transforms::get(sym_index);
    state_history.apply(transform);
    transform->apply(cur_state);
    transform->apply(final_state);
    transform->apply(policy_tensor);
    transform->apply(next_policy_tensor);
  }

  core::seat_index_t cp = cur_state.get_current_player();
  GameOutcome outcome = outcome_;
  eigen_util::left_rotate(outcome, cp);

  InputTensor input_tensor;
  Tensorizor::tensorize(input_tensor, cur_state, state_history);

  for (int k = 0; k < num_keys; ++k) {
    const char* key = keys[k];
    float* value = values[k];

    if (std::strcmp(key, "input") == 0) {
      for (int i = 0; i < input_tensor.size(); ++i) {
        value[i] = input_tensor.data()[i];
      }
      continue;
    }
    if (std::strcmp(key, "policy") == 0) {
      for (int i = 0; i < policy_tensor.size(); ++i) {
        value[i] = policy_tensor.data()[i];
      }
      continue;
    }
    if (std::strcmp(key, "value") == 0) {
      for (int i = 0; i < outcome.size(); ++i) {
        value[i] = outcome[i];
      }
      continue;
    }
    if (std::strcmp(key, "opp_policy") == 0) {
      for (int i = 0; i < next_policy_tensor.size(); ++i) {
        value[i] = next_policy_tensor.data()[i];
      }
      continue;
    }

    bool matched = false;
    constexpr size_t N = mp::Length_v<AuxTargetList>;
    mp::constexpr_for<0, N, 1>([&](auto a) {
      using AuxTarget = mp::TypeAt_t<AuxTargetList, a>;
      if (std::strcmp(key, AuxTarget::kName) == 0) {
        using AuxTensor = typename AuxTarget::Tensor;
        AuxTensor aux_tensor;
        AuxTarget::tensorize(aux_tensor, cur_state, final_state);
        for (int i = 0; i < aux_tensor.size(); ++i) {
          value[i] = aux_tensor.data()[i];
        }
        matched = true;
      }
    });

    if (!matched) {
      throw util::Exception("Unknown key '%s' in file %s", key, filename_.c_str());
    }
  }
  // TODO: load aux data
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
ShapeInfo* GameReadLog<GameState, Tensorizor>::get_shape_info() {
  constexpr int n_base = 4;  // input, policy, value, opp_policy
  constexpr int n_aux = mp::Length_v<AuxTargetList>;
  constexpr int n = n_base + n_aux;

  ShapeInfo* info = new ShapeInfo[n+1];
  info[0].init<InputTensor>("input");
  info[1].init<PolicyTensor>("policy");
  info[2].init<ValueTensor>("value");
  info[3].init<PolicyTensor>("opp_policy");

  mp::constexpr_for<0, n_aux, 1>([&](auto a) {
    using AuxTarget = mp::TypeAt_t<AuxTargetList, a>;
    using AuxTensor = typename AuxTarget::Tensor;
    info[n_base + a].template init<AuxTensor>(AuxTarget::kName);
  });

  return info;
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
void GameReadLog<GameState, Tensorizor>::load_policy(PolicyTensor* policy,
                                                     AuxMemOffset aux_mem_offset) {
  if (aux_mem_offset < 0) {
    policy->setZero();
    return;
  }
  AuxData aux_data;
  int data_offset = aux_data_offset(aux_mem_offset);
  seek_and_read(data_offset, &aux_data);

  int policy_offset = data_offset + sizeof(AuxData);
  auto f = aux_data.policy_target_format;
  if (f < 0) {  // dense format
    seek_and_read(policy_offset, policy);
  } else if (f > 0) {
    sparse_policy_entry_t sparse_entries[f];
    seek_and_read(policy_offset, sparse_entries, f);
    policy->setZero();
    for (int i = 0; i < f; ++i) {
      policy->data()[sparse_entries[i].offset] = sparse_entries[i].probability;
    }
  } else {
    policy->setZero();
  }
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
template<typename T>
void GameReadLog<GameState, Tensorizor>::seek_and_read(int offset, T* data, int count) {
  fseek(file_, offset, SEEK_SET);
  int n = fread(data, sizeof(T), count, file_);
  if (n != count) {
    throw util::Exception("Failed to read data from %s offset=%d (%d != %d)", filename_.c_str(),
                          offset, n, count);
  }
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::outcome_start() const {
  return sizeof(Header);
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::symmetric_index_start() const {
  return outcome_start() + sizeof(GameOutcome);
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::non_symmetric_index_start() const {
  return symmetric_index_start() +
         header_.num_samples_with_symmetry_expansion * sizeof(SymSampleIndex);
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::aux_index_start() const {
  return non_symmetric_index_start() +
         header_.num_samples_without_symmetry_expansion * sizeof(NonSymSampleIndex);
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::state_data_start() const {
  return aux_index_start() + num_aux_data() * sizeof(AuxMemOffset);
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::aux_data_start() const {
  return state_data_start() + header_.num_game_states * sizeof(GameStateData);
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::symmetric_index_offset(int index) const {
  return symmetric_index_start() + index * sizeof(SymSampleIndex);
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::non_symmetric_index_offset(int index) const {
  return non_symmetric_index_start() + index * sizeof(NonSymSampleIndex);
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::aux_index_offset(int index) const {
  return aux_index_start() + index * sizeof(AuxMemOffset);
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::state_data_offset(int index) const {
  return state_data_start() + index * sizeof(GameStateData);
}

template <concepts::Game Game, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::aux_data_offset(int mem_offset) const {
  return aux_data_start() + mem_offset;
}

}  // namespace core
