#include <core/GameLog.hpp>

#include <util/BitSet.hpp>
#include <util/EigenUtil.hpp>

#include <limits>

namespace core {

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
GameWriteLog<GameState, Tensorizor>::GameWriteLog(game_id_t id, int64_t start_timestamp)
    : id_(id), start_timestamp_(start_timestamp) {}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
GameWriteLog<GameState, Tensorizor>::~GameWriteLog() {
  for (Entry* entry : entries_) {
    delete entry;
  }
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void GameWriteLog<GameState, Tensorizor>::add(const GameState& state, const Action& action,
                                              const PolicyTensor* policy_target,
                                              bool use_for_training) {
  // TODO: get entries from a thread-specific object pool
  Entry* entry = new Entry();
  entry->state_data = state.data();
  entry->symmetries = state.get_symmetry_indices();
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
  }
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void GameWriteLog<GameState, Tensorizor>::add_terminal(const GameState& state,
                                                       const GameOutcome& outcome) {
  if (terminal_added_) return;
  terminal_added_ = true;
  Entry* entry = new Entry();
  entry->state_data = state.data();
  entry->policy_target.setZero();
  GameStateTypes::nullify_action(entry->action);
  entry->use_for_training = false;
  entry->policy_target_is_valid = false;
  entry->terminal = true;
  entries_.push_back(entry);

  outcome_ = outcome;
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void GameWriteLog<GameState, Tensorizor>::serialize(std::ostream& stream) const {
  using GameReadLog = core::GameReadLog<GameState, Tensorizor>;
  using Header = typename GameReadLog::Header;
  using SymSampleIndex = typename GameReadLog::SymSampleIndex;
  using NonSymSampleIndex = typename GameReadLog::NonSymSampleIndex;
  using AuxMemOffset = typename GameReadLog::AuxMemOffset;
  using AuxData = typename GameReadLog::AuxData;
  using sparse_policy_entry_t = typename GameReadLog::sparse_policy_entry_t;

  Header header;
  std::ostringstream state_data_stream;
  std::ostringstream aux_data_stream;

  std::vector<SymSampleIndex> sym_sample_indices;
  std::vector<NonSymSampleIndex> non_sym_sample_indices;
  std::vector<AuxMemOffset> aux_mem_offsets;

  util::release_assert(!entries_.empty(), "Illegal serialization of empty GameWriteLog");

  for (const Entry* entry : entries_) {
    uint32_t move_num = header.num_game_states;
    const GameStateData& state_data = entry->state_data;

    state_data_stream.write(reinterpret_cast<const char*>(&state_data), sizeof(state_data));

    if (entry->use_for_training) {
      for (symmetry_index_t sym_index : bitset_util::on_indices(entry->symmetries)) {
        sym_sample_indices.emplace_back(move_num, sym_index);
      }
      non_sym_sample_indices.emplace_back(move_num);
      header.num_samples_with_symmetry_expansion += entry->symmetries.count();
      header.num_samples_without_symmetry_expansion++;
    }

    AuxMemOffset aux_mem_offset = aux_data_stream.tellp();
    aux_mem_offsets.push_back(aux_mem_offset);

    char policy_target_buf[sizeof(PolicyTensor)];
    char* policy_target_buf_ptr = policy_target_buf;
    int16_t format = write_policy_target(*entry, &policy_target_buf_ptr);

    header.num_game_states++;

    if (entry->terminal) continue;

    AuxData aux_data;
    aux_data.action = entry->action;
    aux_data.policy_target_format = format;
    aux_data.use_for_training = entry->use_for_training;

    aux_data_stream.write(reinterpret_cast<const char*>(&aux_data), sizeof(aux_data));
    if (format < 0) {
      aux_data_stream.write(policy_target_buf, sizeof(PolicyTensor));
    } else if (format > 0) {
      aux_data_stream.write(policy_target_buf, sizeof(sparse_policy_entry_t) * format);
    }

  }

  stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
  stream.write(reinterpret_cast<const char*>(&outcome_), sizeof(outcome_));
  stream.write(reinterpret_cast<const char*>(sym_sample_indices.data()),
               sizeof(SymSampleIndex) * sym_sample_indices.size());
  stream.write(reinterpret_cast<const char*>(non_sym_sample_indices.data()),
               sizeof(NonSymSampleIndex) * non_sym_sample_indices.size());
  stream.write(reinterpret_cast<const char*>(aux_mem_offsets.data()),
               sizeof(AuxMemOffset) * aux_mem_offsets.size());
  stream.write(state_data_stream.str().data(), state_data_stream.tellp());
  stream.write(aux_data_stream.str().data(), aux_data_stream.tellp());
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
bool GameWriteLog<GameState, Tensorizor>::is_previous_entry_used_for_training() const {
  if (entries_.empty()) {
    return false;
  }
  return entries_.back()->use_for_training;
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int16_t GameWriteLog<GameState, Tensorizor>::write_policy_target(const Entry& entry,
                                                                 char** buffer) {
  using GameReadLog = core::GameReadLog<GameState, Tensorizor>;
  using sparse_policy_entry_t = typename GameReadLog::sparse_policy_entry_t;

  if (!entry.policy_target_is_valid) {
    return 0;
  }

  const PolicyTensor& policy_target = entry.policy_target;
  int num_nonzero_entries = eigen_util::count(policy_target);

  int sparse_repr_size = sizeof(sparse_policy_entry_t) * num_nonzero_entries;
  int dense_repr_size = sizeof(PolicyTensor);

  if (sparse_repr_size * 2 > dense_repr_size) {
    // use dense representation
    std::memcpy(*buffer, policy_target.data(), dense_repr_size);
    return -1;
  }

  constexpr int N = eigen_util::extract_shape_t<PolicyTensor>::total_size;
  const auto* data = policy_target.data();
  for (int i = 0; i < N; ++i) {
    if (data[i]) {
      sparse_policy_entry_t entry{i, data[i]};
      std::memcpy(*buffer, &entry, sizeof(entry));
      *buffer += sizeof(entry);
    }
  }

  if (num_nonzero_entries > std::numeric_limits<int16_t>::max()) {
    throw util::Exception("Too many non-zero entries in policy target (%d)", num_nonzero_entries);
  }
  return num_nonzero_entries;
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
GameReadLog<GameState, Tensorizor>::GameReadLog(const char* filename) {
  filename_ = filename;
  file_ = fopen(filename, "rb");
  util::release_assert(file_, "Failed to open file '%s' for reading", filename);

  if (fread(&header_, sizeof(Header), 1, file_) != sizeof(Header)) {
    fclose(file_);
    file_ = nullptr;
    throw util::Exception("Failed to read header from file '%s'", filename);
  }

  if (fread(&outcome_, sizeof(GameOutcome), 1, file_) != sizeof(GameOutcome)) {
    fclose(file_);
    file_ = nullptr;
    throw util::Exception("Failed to read outcome from file '%s'", filename);
  }
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
GameReadLog<GameState, Tensorizor>::~GameReadLog() {
  if (file_) fclose(file_);
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void GameReadLog<GameState, Tensorizor>::load(int index, bool apply_symmetry, float* input,
                                              const char** keys, float** values, int num_keys) {
  util::release_assert(file_, "Attempt to read from closed GameReadLog");

  if (index < 0 || index >= (int)header_.num_game_states) {
    throw util::Exception("Index %d out of bounds in GameReadLog (%u)", index,
                          header_.num_game_states);
  }

  int state_index;
  core::symmetry_index_t sym_index = -1;

  if (apply_symmetry) {
    SymSampleIndex sym_sample_index;
    seek_and_read(symmetric_index_offset(index), &sym_sample_index);

    state_index = sym_sample_index.state_index;
    sym_index = sym_sample_index.symmetry_index;
  } else {
    NonSymSampleIndex non_sym_sample_index;
    seek_and_read(non_symmetric_index_offset(index), &non_sym_sample_index);

    state_index = non_sym_sample_index;
  }

  AuxMemOffset aux_mem_offset;
  seek_and_read(aux_index_offset(index), &aux_mem_offset);

  AuxMemOffset next_aux_mem_offset = -1;
  if (index + 1 < (int)header_.num_game_states) {
    seek_and_read(aux_index_offset(index + 1), &next_aux_mem_offset);
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

  for (int i = 0; i < input_tensor.size(); ++i) {
    input[i] = input_tensor.data()[i];
  }

  for (int k = 0; k < num_keys; ++k) {
    const char* key = keys[k];
    float* value = values[k];
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
    mp::constexpr_for<0, N, 1>([&](auto i) {
      using AuxTarget = mp::TypeAt_t<AuxTargetList, i>;
      if (std::strcmp(key, AuxTarget::kName) == 0) {
        using AuxTensor = typename AuxTarget::Tensor;
        AuxTensor aux_tensor;
        AuxTarget::tensorize(aux_tensor, cur_state, final_state);
        matched = true;
      }
    });

    if (!matched) {
      throw util::Exception("Unknown key '%s' in file %s", key, filename_.c_str());
    }
  }
  // TODO: load aux data
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void GameReadLog<GameState, Tensorizor>::load_policy(PolicyTensor* policy,
                                                     AuxMemOffset aux_mem_offset) {
  if (aux_mem_offset < 0) {
    policy->setZero();
    return;
  }
  AuxData aux_data;
  seek_and_read(aux_mem_offset, &aux_data);

  auto f = aux_data.policy_target_format;
  if (f < 0) {  // dense format
    seek_and_read(aux_data_offset(aux_mem_offset), policy);
  } else if (f > 0) {
    sparse_policy_entry_t sparse_entries[f];
    seek_and_read(aux_data_offset(aux_mem_offset), sparse_entries, f);
    policy->setZero();
    for (int i = 0; i < f; ++i) {
      policy->data()[sparse_entries[i].offset] = sparse_entries[i].probability;
    }
  } else {
    policy->setZero();
  }
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
template<typename T>
void GameReadLog<GameState, Tensorizor>::seek_and_read(int offset, T* data, int count) {
  fseek(file_, offset, SEEK_SET);
  constexpr int t = sizeof(T);
  int n = fread(data, t, count, file_);
  if (n != count * t) {
    throw util::Exception("Failed to read data from GameReadLog (%d != %d * %d)", n, count, t);
  }
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::outcome_start() const {
  return sizeof(Header);
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::symmetric_index_start() const {
  return outcome_start() + sizeof(GameOutcome);
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::non_symmetric_index_start() const {
  return symmetric_index_start() +
         header_.num_samples_with_symmetry_expansion * sizeof(SymSampleIndex);
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::aux_index_start() const {
  return non_symmetric_index_start() +
         header_.num_samples_without_symmetry_expansion * sizeof(NonSymSampleIndex);
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::state_data_start() const {
  return aux_index_start() + num_aux_data() * sizeof(AuxMemOffset);
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::aux_data_start() const {
  return state_data_start() + header_.num_game_states * sizeof(GameStateData);
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::symmetric_index_offset(int index) const {
  return symmetric_index_start() + index * sizeof(SymSampleIndex);
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::non_symmetric_index_offset(int index) const {
  return non_symmetric_index_start() + index * sizeof(NonSymSampleIndex);
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::aux_index_offset(int index) const {
  return aux_index_start() + index * sizeof(AuxMemOffset);
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::state_data_offset(int index) const {
  return state_data_start() + index * sizeof(GameStateData);
}

template <GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int GameReadLog<GameState, Tensorizor>::aux_data_offset(int mem_offset) const {
  return aux_data_start() + mem_offset;
}

}  // namespace core
