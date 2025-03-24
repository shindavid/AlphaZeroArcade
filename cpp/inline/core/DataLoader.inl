#include <core/DataLoader.hpp>

#include <core/GameLog.hpp>
#include <cstdio>
#include <util/Asserts.hpp>
#include <util/Exception.hpp>
#include <util/Random.hpp>

#include <boost/filesystem.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <mutex>

namespace core {

template <concepts::Game Game>
DataLoader<Game>::DataFile::DataFile(const char* filename, int gen, int num_rows, int64_t file_size)
    : filename_(filename), gen_(gen), num_rows_(num_rows), file_size_(file_size) {
  printf("%p %s\n", this, std::format("DataFile::{}({})", __func__, filename_).c_str());
}

template <concepts::Game Game>
DataLoader<Game>::DataFile::~DataFile() {
  printf("%s\n", std::format("DataFile::{}() - {} - {}", __func__, filename_, __LINE__).c_str());
  unload();
  printf("%s\n", std::format("DataFile::{}() - {} - {}", __func__, filename_, __LINE__).c_str());
}

template <concepts::Game Game>
void DataLoader<Game>::DataFile::load() {
  std::unique_lock lock(mutex_);
  if (!buffer_) {
    printf("%s\n", std::format("DataFile::{}({})", __func__, filename_).c_str());
    FILE* file = fopen(filename_.c_str(), "rb");
    if (!file) {
      throw util::Exception("Failed to open file '%s'", filename_.c_str());
    }

    if (fseek(file, 0, SEEK_SET) != 0) {
      throw util::Exception("Failed to seek to start of file '%s'", filename_.c_str());
    }

    buffer_ = new char[file_size_];
    int64_t read_size = fread(buffer_, 1, file_size_, file);
    if (read_size != file_size_) {
      throw util::Exception("Failed to read data from file '%s'", filename_.c_str());
    }

    fclose(file);
  }
  printf("%s\n", std::format("DataFile::{}({}) - complete", __func__, filename_).c_str());
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
int64_t DataLoader<Game>::DataFile::unload() {
  std::unique_lock lock(mutex_);
  if (!buffer_) return 0;

  delete[] buffer_;
  buffer_ = nullptr;

  lock.unlock();
  cv_.notify_all();
  return file_size_;
}

template <concepts::Game Game>
const char* DataLoader<Game>::DataFile::buffer() const {
  std::unique_lock lock(mutex_);
  printf("%s\n", std::format("DataFile::{}() - {} - {}", __func__, filename_, __LINE__).c_str());
  cv_.wait(lock, [this] { return buffer_ != nullptr; });
  printf("%s\n", std::format("DataFile::{}() - {} - {}", __func__, filename_, __LINE__).c_str());
  return buffer_;
}

template <concepts::Game Game>
void DataLoader<Game>::LoadInstructions::init(bool apply_sym, int n_targets, float* output_array,
                                              int* target_indices_array) {
  apply_symmetry = apply_sym;
  output_data_array = output_array;
  target_indices.resize(n_targets);
  for (int i = 0; i < n_targets; ++i) {
    target_indices[i] = target_indices_array[i];
  }

  row_size = Game::InputTensorizor::Tensor::Dimensions::total_size;

  using TrainingTargetsList = Game::TrainingTargets::List;
  constexpr size_t N = mp::Length_v<TrainingTargetsList>;

  for (int target_index : target_indices) {
    mp::constexpr_for<0, N, 1>([&](auto a) {
      if (target_index == a) {
        using Target = mp::TypeAt_t<TrainingTargetsList, a>;
        constexpr int kSize = Target::Tensor::Dimensions::total_size;
        row_size += kSize;
        row_size++;  // for the mask
      }
    });
  }
}

template <concepts::Game Game>
DataLoader<Game>::SamplingManager::~SamplingManager() {
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
  for (local_index_vec_t* vec : vec_pool_used_) {
    delete vec;
  }
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
  for (local_index_vec_t* vec : vec_pool_unused_) {
    delete vec;
  }
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
}

template <concepts::Game Game>
void DataLoader<Game>::SamplingManager::sample(work_unit_deque_t* work_units,
                                               const file_deque_t& files, int64_t window_size,
                                               int n_samples) {
  reset_vec_pools();
  sampled_indices_.resize(n_samples);
  printf("%s\n", std::format("{}(..., {}, {})", __func__, window_size, n_samples).c_str());
  for (int i = 0; i < n_samples; ++i) {
    sampled_indices_[i] = util::Random::uniform_sample(0, window_size);
  }
  std::sort(sampled_indices_.begin(), sampled_indices_.end());

  int index = 0;
  int64_t offset = 0;
  generation_t gen = -1;
  for (DataFile* file : files) {
    util::release_assert(gen == -1 || file->gen() < gen, "DataFileSet::files() bug");
    gen = file->gen();
    int64_t num_rows = file->num_rows();
    int64_t limit = offset + num_rows;
    int64_t local_offset = std::max(limit - window_size, int64_t(0));
    int output_index = index;

    // Note: it important to create a WorkUnit even if the gen has no samples, because WorkManager
    // detects when a DataFile is safe to remove based on the smallest gen in work_units.
    local_index_vec_t* local_indices = get_vec();
    while (index < n_samples && sampled_indices_[index] < limit) {
      int64_t local_index = local_offset + sampled_indices_[index] - offset;
      util::release_assert(local_index >= 0 && local_index < num_rows,
                           "SamplingManager::sample() bug at %d [local_index:%ld num_rows:%ld]",
                           __LINE__, local_index, num_rows);
      local_indices->push_back(local_index);
      index++;
    }
    work_units->push_back(WorkUnit{file, local_indices, output_index});
    offset = limit;
    if (offset >= window_size) {
      sampled_indices_.clear();
      return;
    }
  }
  throw util::Exception("SamplingManager::sample() bug at %d [window_size:%ld offset:%ld files:%ld]",
                        __LINE__, window_size, offset, files.size());
}

template <concepts::Game Game>
typename DataLoader<Game>::local_index_vec_t* DataLoader<Game>::SamplingManager::get_vec() {
  local_index_vec_t* local_indices;
  if (!vec_pool_unused_.empty()) {
    local_indices = vec_pool_unused_.back();
    vec_pool_unused_.pop_back();
  } else {
    local_indices = new local_index_vec_t();
  }
  vec_pool_used_.push_back(local_indices);
  return local_indices;
}

template <concepts::Game Game>
void DataLoader<Game>::SamplingManager::reset_vec_pools() {
  for (local_index_vec_t* vec : vec_pool_used_) {
    vec->clear();
    vec_pool_unused_.push_back(vec);
  }
  vec_pool_used_.clear();
}

template <concepts::Game Game>
DataLoader<Game>::ThreadTable::ThreadTable(int n_threads) : n_threads_(n_threads) {
  for (int i = 0; i < n_threads; ++i) {
    available_thread_ids_.push_back(i);
  }
}

template <concepts::Game Game>
void DataLoader<Game>::ThreadTable::mark_as_available(thread_id_t id) {
  std::unique_lock lock(mutex_);
  available_thread_ids_.push_back(id);
  lock.unlock();
  cv_.notify_one();
}

template <concepts::Game Game>
typename DataLoader<Game>::thread_id_t DataLoader<Game>::ThreadTable::allocate_thread() {
  std::unique_lock lock(mutex_);
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
  cv_.wait(lock, [this] { return quitting_ || !available_thread_ids_.empty(); });
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
  if (quitting_) return -1;
  thread_id_t id = available_thread_ids_.back();
  available_thread_ids_.pop_back();
  return id;
}

template <concepts::Game Game>
void DataLoader<Game>::ThreadTable::wait_until_all_threads_available() {
  std::unique_lock lock(mutex_);
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
  cv_.wait(lock, [this] {
    return quitting_ || (int)available_thread_ids_.size() == n_threads_;
    });
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
}

template <concepts::Game Game>
void DataLoader<Game>::ThreadTable::quit() {
  std::unique_lock lock(mutex_);
  quitting_ = true;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
DataLoader<Game>::PrefetchThread::PrefetchThread(ThreadTable* table, thread_id_t id)
    : table_(table), id_(id) {
  thread_ = std::thread(&PrefetchThread::loop, this);
}

template <concepts::Game Game>
DataLoader<Game>::PrefetchThread::~PrefetchThread() {
  printf("%p %s\n", this, std::format("{}() - {}", __func__, __LINE__).c_str());
  quit();
  printf("%p %s\n", this, std::format("{}() - {}", __func__, __LINE__).c_str());
}

template <concepts::Game Game>
void DataLoader<Game>::PrefetchThread::quit() {
  printf("%p %s\n", this, std::format("{}() - {}", __func__, __LINE__).c_str());
  std::unique_lock lock(mutex_);
  printf("%p %s\n", this, std::format("{}() - {}", __func__, __LINE__).c_str());
  quitting_ = true;
  lock.unlock();
  cv_.notify_all();
  printf("%p %s\n", this, std::format("{}() - {}", __func__, __LINE__).c_str());
  thread_.join();
  printf("%p %s\n", this, std::format("{}() - {}", __func__, __LINE__).c_str());
}

template <concepts::Game Game>
void DataLoader<Game>::PrefetchThread::schedule_prefetch(DataFile* data_file) {
  std::unique_lock lock(mutex_);
  file_ = data_file;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void DataLoader<Game>::PrefetchThread::loop() {
  while (!quitting_) {
    printf("%p %s\n", this, std::format("{}() - {}", __func__, __LINE__).c_str());
    std::unique_lock lock(mutex_);
    printf("%p %s\n", this, std::format("{}() - {}", __func__, __LINE__).c_str());
    cv_.wait(lock, [this] { return quitting_ || file_ != nullptr; });
    printf("%p %s\n", this, std::format("{}() - {}", __func__, __LINE__).c_str());
    if (quitting_) return;
    lock.unlock();

    file_->load();
    file_ = nullptr;
    table_->mark_as_available(id_);
  }
}

template <concepts::Game Game>
DataLoader<Game>::FileManager::FileManager(const boost::filesystem::path& data_dir,
                                           int64_t memory_budget, int num_prefetch_threads)
    : data_dir_(data_dir), memory_budget_(memory_budget), thread_table_(num_prefetch_threads) {
  printf("%p %s\n", this, std::format("{}() - data_dir_={}", __func__, data_dir_.c_str()).c_str());
  for (int i = 0; i < num_prefetch_threads; ++i) {
    prefetch_threads_.push_back(new PrefetchThread(&thread_table_, i));
  }
  prefetch_loop_thread_ = std::thread(&FileManager::prefetch_loop, this);
}

template <concepts::Game Game>
DataLoader<Game>::FileManager::~FileManager() {
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
  thread_table_.quit();
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
  exit_prefetch_loop();
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
  delete_all_files();
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());

  for (PrefetchThread* thread : prefetch_threads_) {
    delete thread;
  }
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
  prefetch_loop_thread_.join();
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
}

template <concepts::Game Game>
int DataLoader<Game>::FileManager::prepare_files(const work_unit_deque_t& work_units) {
  std::unique_lock lock(mutex_);
  load_queue_.clear();
  unload_queue_.clear();
  active_file_count_ = 0;

  int64_t expected_memory_usage = 0;
  generation_t start_gen = -1;
  for (const WorkUnit& unit : work_units) {
    DataFile* file = unit.file;
    util::release_assert(start_gen == -1 || file->gen() < start_gen,
                         "DataFileSet::prepare_files() bug at %d", __LINE__);
    start_gen = file->gen();
    if (!unit.sample_indices->empty()) {
      if (file->is_loaded()) {
        active_file_count_++;
        expected_memory_usage += file->file_size();
      } else {
        load_queue_.push_back(file);
      }
    }
  }

  if (start_gen >= 0) {
    trim(start_gen);
  }

  util::release_assert(memory_usage_ == expected_memory_usage,
    "DataFileSet::prepare_files() memory-usage-tracking-bug [%ld != %ld]",
     memory_usage_, expected_memory_usage);

  lock.unlock();
  cv_.notify_all();
  return start_gen;
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::restore(int n, generation_t* gens, int* row_counts,
                                            int64_t* file_sizes) {
  util::release_assert(all_files_.empty(), "FileManager::init() bug");

  for (int i = 0; i < n; ++i) {
    append(gens[i], row_counts[i], file_sizes[i]);
  }
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::append(int end_gen, int num_rows, int64_t file_size) {
  auto filename = data_dir_ / std::format("gen-{}.data", end_gen);
  DataFile* data_file = new DataFile(filename.c_str(), end_gen, num_rows, file_size);

  std::unique_lock lock(mutex_);
  all_files_.push_front(data_file);
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::decrement_active_file_count() {
  std::unique_lock lock(mutex_);
  util::release_assert(active_file_count_ > 0, "DataFileSet::decrement_active_file_count() bug");
  active_file_count_--;
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::trim(int start_gen) {
  // Unload all files with gen < start_gen
  while (!all_files_.empty() && all_files_.back()->gen() < start_gen) {
    DataFile* file = all_files_.back();
    memory_usage_ -= file->unload();
    delete file;
    all_files_.pop_back();
  }
}

template <concepts::Game Game>
typename DataLoader<Game>::FileManager::Instruction
DataLoader<Game>::FileManager::get_next_instruction() const {
  if (quitting_) return kQuit;

  if (load_queue_.empty()) {
    // Nothing left to prefetch
    return kWait;
  }

  // Something is available to prefetch
  DataFile* file = load_queue_.front();
  util::release_assert(!file->is_loaded(), "DataFileSet::prefetch_loop() bug at %d", __LINE__);

  if (memory_usage_ + file->file_size() <= memory_budget_) {
    // We have sufficient memory
    return kLoad;
  }

  // We don't have enough memory

  if (!unload_queue_.empty()) {
    // We have some files that we can unload
    return kUnload;
  }

  if (active_file_count_) {
    // We can wait for these to finish before prefetching more
    return kWait;
  }

  util::release_assert(file->file_size() > memory_budget_, "DataFileSet::prefetch_loop() bug at %d",
                       __LINE__);

  // our memory budget is insufficient to load this single file. In this case let's just violate
  // the memory budget and load the file anyway
  return kLoad;
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::prefetch_loop() {
  while (!quitting_) {
    std::unique_lock lock(mutex_);
    Instruction instruction = kWait;
    printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
    cv_.wait(lock, [&] {
      instruction = get_next_instruction();
      return instruction != kWait;
    });
    printf("%s\n", std::format("{}() - {} - {}", __func__, (int)instruction, __LINE__).c_str());

    if (instruction == kQuit) {
      return;
    } else if (instruction == kLoad) {
      DataFile* file = load_queue_.front();

      thread_id_t id = thread_table_.allocate_thread();
      prefetch_threads_[id]->schedule_prefetch(file);
      memory_usage_ += file->file_size();

      load_queue_.pop_front();
      active_file_count_++;
    } else if (instruction == kUnload) {
      DataFile* file = unload_queue_.front();
      memory_usage_ -= file->unload();
      unload_queue_.pop_front();
    } else {
      throw util::Exception("DataFileSet::prefetch_loop() bug at %d", __LINE__);
    }
  }
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::exit_prefetch_loop() {
  std::unique_lock lock(mutex_);
  quitting_ = true;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::delete_all_files() {
  for (DataFile* file : all_files_) {
    delete file;
  }
  all_files_.clear();
}

template <concepts::Game Game>
DataLoader<Game>::WorkerThread::WorkerThread(FileManager* file_manager, ThreadTable* table,
                                             thread_id_t id)
    : file_manager_(file_manager), table_(table), id_(id) {
  thread_ = std::thread(&WorkerThread::loop, this);
}

template <concepts::Game Game>
DataLoader<Game>::WorkerThread::~WorkerThread() {
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
  quit();
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
}

template <concepts::Game Game>
void DataLoader<Game>::WorkerThread::quit() {
  std::unique_lock lock(mutex_);
  quitting_ = true;
  lock.unlock();
  cv_.notify_all();
  thread_.join();
}

template <concepts::Game Game>
void DataLoader<Game>::WorkerThread::schedule_work(const LoadInstructions& load_instructions,
                                                   const WorkUnit& unit) {
  std::unique_lock lock(mutex_);
  load_instructions_ = &load_instructions;
  unit_ = unit;
  has_work_ = true;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void DataLoader<Game>::WorkerThread::loop() {
  while (!quitting_) {
    std::unique_lock lock(mutex_);
    printf("%s\n", std::format("WorkerThread::{}() - {}", __func__, __LINE__).c_str());
    cv_.wait(lock, [this] { return quitting_ || has_work_; });
    printf("%s\n", std::format("WorkerThread::{}() - {}", __func__, __LINE__).c_str());
    if (quitting_) return;

    do_work();
    has_work_ = false;
    table_->mark_as_available(id_);
    lock.unlock();

    file_manager_->decrement_active_file_count();
  }
}

template <concepts::Game Game>
void DataLoader<Game>::WorkerThread::do_work() {
  const local_index_vec_t& sample_indices = *unit_.sample_indices;

  int num_samples = sample_indices.size();
  util::release_assert(num_samples > 0, "WorkerThread::do_work() bug at %d", __LINE__);

  DataFile* file = unit_.file;
  int output_index = unit_.output_index;

  const char* filename = file->filename();
  printf("WorkerThread::%s() - filename=%s output_index=%d num_samples=%d\n", __func__, filename, output_index,
         num_samples);
  std::cout.flush();
  const char* buffer = file->buffer();  // blocks until loaded

  bool apply_symmetry = load_instructions_->apply_symmetry;
  float* output_data_array = load_instructions_->output_data_array;
  const std::vector<int>& target_indices = load_instructions_->target_indices;
  int row_size = load_instructions_->row_size;

  GameLogFileReader reader(buffer);

  int num_games = reader.num_games();

  int offset = 0;
  int s = 0;
  for (int g = 0; g < num_games; ++g) {
    int limit = offset + reader.num_samples(g);
    if (sample_indices[s] < limit) {
      GameReadLog log(filename, g, reader.metadata(g), reader.game_data_buffer(g));

      for (; s < num_samples && sample_indices[s] < limit; ++s) {
        int row_index = sample_indices[s] - offset;
        float* output = output_data_array + output_index * row_size;
        log.load(row_index, apply_symmetry, target_indices, output);
        output_index++;
      }

      if (s == num_samples) return;
    }
    offset = limit;
  }

  throw util::Exception("WorkerThread::do_work() bug at %d (%d != %d) num_games=%d", __LINE__, s,
                        num_samples, num_games);
}

template <concepts::Game Game>
DataLoader<Game>::WorkManager::WorkManager(FileManager* file_manager, int num_threads)
    : thread_table_(num_threads) {
  for (int i = 0; i < num_threads; ++i) {
    workers_.push_back(new WorkerThread(file_manager, &thread_table_, i));
  }
}

template <concepts::Game Game>
DataLoader<Game>::WorkManager::~WorkManager() {
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
  thread_table_.quit();
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
  for (WorkerThread* thread : workers_) {
    delete thread;
  }
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
}

template <concepts::Game Game>
void DataLoader<Game>::WorkManager::process(const LoadInstructions& load_instructions,
                                            work_unit_deque_t& work_units) {
  while (!work_units.empty()) {
    thread_id_t id = thread_table_.allocate_thread();
    WorkerThread* worker = workers_[id];
    worker->schedule_work(load_instructions, work_units.front());
    work_units.pop_front();
  }

  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
  thread_table_.wait_until_all_threads_available();
  printf("%s\n", std::format("{}() - {}", __func__, __LINE__).c_str());
}

template <concepts::Game Game>
DataLoader<Game>::DataLoader(const Params& params)
    : params_(params),
      file_manager_(params.data_dir, params.memory_budget, params.num_prefetch_threads),
      work_manager_(&file_manager_, params.num_worker_threads) {}

template <concepts::Game Game>
void DataLoader<Game>::restore(int n, generation_t* gens, int* row_counts, int64_t* file_sizes) {
  printf("%s\n",
         std::format("DataLoader::{}({}, {}...{})", __func__, n, gens[0], gens[n - 1]).c_str());

  file_manager_.restore(n, gens, row_counts, file_sizes);
}

template <concepts::Game Game>
void DataLoader<Game>::add_gen(int gen, int num_rows, int64_t file_size) {
  printf("%s\n",
         std::format("DataLoader::{}({}, {}, {})", __func__, gen, num_rows, file_size).c_str());

  file_manager_.append(gen, num_rows, file_size);
}

template <concepts::Game Game>
void DataLoader<Game>::load(int64_t window_size, int n_samples, bool apply_symmetry, int n_targets,
                            float* output_array, int* target_indices_array, int* start_gen) {
  load_instructions_.init(apply_symmetry, n_targets, output_array, target_indices_array);
  sampling_manager_.sample(&work_units_, file_manager_.files_in_reverse_order(), window_size,
                           n_samples);
  for (const auto& unit : work_units_) {
    printf("%s\n", std::format("DBG unit {} gen={} num_samples={} output={}",
                               unit.file->filename(), unit.file->gen(), unit.sample_indices->size(),
                               unit.output_index).c_str());
  }
  start_gen[0] = file_manager_.prepare_files(work_units_);
  work_manager_.process(load_instructions_, work_units_);
  shuffle_output(n_samples);
}

template <concepts::Game Game>
void DataLoader<Game>::shuffle_output(int n_samples) {
  float* f = load_instructions_.output_data_array;
  int row_size = load_instructions_.row_size;
  util::Random::chunked_shuffle(f, f + row_size * n_samples, row_size);
}

}  // namespace core
