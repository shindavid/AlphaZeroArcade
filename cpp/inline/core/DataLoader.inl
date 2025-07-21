#include "core/DataLoader.hpp"

#include "core/GameLog.hpp"
#include "util/Asserts.hpp"
#include "util/Exceptions.hpp"
#include "util/FileUtil.hpp"
#include "util/IndexedDispatcher.hpp"
#include "util/Random.hpp"
#include "util/mit/mit.hpp"

#include <boost/filesystem.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>

namespace core {

template <concepts::Game Game>
DataLoader<Game>::DataFile::DataFile(const char* filename, int gen, int num_rows, int64_t file_size)
    : filename_(filename), gen_(gen), num_rows_(num_rows), file_size_(file_size) {}

template <concepts::Game Game>
DataLoader<Game>::DataFile::~DataFile() {
  unload();
}

template <concepts::Game Game>
void DataLoader<Game>::DataFile::load() {
  mit::unique_lock lock(mutex_);
  if (!buffer_) {
    buffer_ = util::read_file(filename_.c_str(), file_size_);
  }
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
int64_t DataLoader<Game>::DataFile::unload() {
  mit::unique_lock lock(mutex_);
  if (!buffer_) return 0;

  delete[] buffer_;
  buffer_ = nullptr;

  lock.unlock();
  cv_.notify_all();
  return file_size_;
}

template <concepts::Game Game>
const char* DataLoader<Game>::DataFile::buffer() const {
  mit::unique_lock lock(mutex_);
  cv_.wait(lock, [this] { return buffer_ != nullptr; });
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
    util::IndexedDispatcher<N>::call(target_index, [&](auto t) {
      using Target = mp::TypeAt_t<TrainingTargetsList, t>;
      using Tensor = Target::Tensor;
      constexpr int kSize = Tensor::Dimensions::total_size;
      row_size += kSize;
      row_size++;  // for the mask
    });
  }
}

template <concepts::Game Game>
DataLoader<Game>::SamplingManager::~SamplingManager() {
  for (local_index_vec_t* vec : vec_pool_used_) {
    delete vec;
  }
  for (local_index_vec_t* vec : vec_pool_unused_) {
    delete vec;
  }
}

template <concepts::Game Game>
void DataLoader<Game>::SamplingManager::sample(work_unit_deque_t* work_units,
                                               const file_deque_t& files, int64_t window_start,
                                               int64_t window_end, int64_t n_total_rows,
                                               int n_samples) {
  reset_vec_pools();
  sampled_indices_.resize(n_samples);
  for (int i = 0; i < n_samples; ++i) {
    sampled_indices_[i] = util::Random::uniform_sample(window_start, window_end);
  }
  std::sort(sampled_indices_.begin(), sampled_indices_.end(), std::greater{});

  int sample_index = 0;
  int64_t file_end = n_total_rows;
  generation_t gen = -1;
  for (DataFile* file : files) {
    RELEASE_ASSERT(gen == -1 || file->gen() < gen, "DataFileSet::files() bug");
    gen = file->gen();
    int64_t num_rows = file->num_rows();
    int64_t file_start = file_end - num_rows;

    if (file_start >= window_end) {
      // Usually, window_end corresponds to the end of the latest file, so it shouldn't be possible
      // to reach this point. The exception is when we are retraining over prior training windows
      // for forked runs.
      file_end = file_start;
      continue;
    }

    // Note: it important to create a WorkUnit even if the gen has no samples, because WorkManager
    // detects when a DataFile is safe to remove based on the smallest gen in work_units.
    local_index_vec_t* local_indices = get_vec();

    int output_index = sample_index;
    while (sample_index < n_samples && sampled_indices_[sample_index] >= file_start) {
      int64_t local_index = sampled_indices_[sample_index] - file_start;
      RELEASE_ASSERT(local_index >= 0 && local_index < num_rows,
                     "SamplingManager::sample() bug at {} [local_index:{} num_rows:{}]", __LINE__,
                     local_index, num_rows);
      local_indices->push_back(local_index);
      sample_index++;
    }

    // NOTE: if we're careful, we can avoid reversing local_indices here. But that would make the
    // logic less clear.
    std::reverse(local_indices->begin(), local_indices->end());

    work_units->push_back(WorkUnit{file, local_indices, output_index});

    file_end = file_start;
    if (file_end <= window_start) {
      sampled_indices_.clear();
      return;
    }
  }
  throw util::Exception("SamplingManager::sample() bug at {} [{}:{}] total:{} sam:{} files:{}",
                        __LINE__, window_start, window_end, n_total_rows, n_samples, files.size());
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
  mit::unique_lock lock(mutex_);
  available_thread_ids_.push_back(id);
  lock.unlock();
  cv_.notify_one();
}

template <concepts::Game Game>
typename DataLoader<Game>::thread_id_t DataLoader<Game>::ThreadTable::allocate_thread() {
  mit::unique_lock lock(mutex_);
  cv_.wait(lock, [this] { return quitting_ || !available_thread_ids_.empty(); });
  if (quitting_) return -1;
  thread_id_t id = available_thread_ids_.back();
  available_thread_ids_.pop_back();
  return id;
}

template <concepts::Game Game>
void DataLoader<Game>::ThreadTable::wait_until_all_threads_available() {
  mit::unique_lock lock(mutex_);
  cv_.wait(lock, [this] { return quitting_ || (int)available_thread_ids_.size() == n_threads_; });
}

template <concepts::Game Game>
void DataLoader<Game>::ThreadTable::quit() {
  mit::unique_lock lock(mutex_);
  quitting_ = true;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
DataLoader<Game>::PrefetchThread::PrefetchThread(ThreadTable* table, thread_id_t id)
    : table_(table), id_(id) {
  thread_ = mit::thread(&PrefetchThread::loop, this);
}

template <concepts::Game Game>
DataLoader<Game>::PrefetchThread::~PrefetchThread() {
  quit();
}

template <concepts::Game Game>
void DataLoader<Game>::PrefetchThread::quit() {
  mit::unique_lock lock(mutex_);
  quitting_ = true;
  lock.unlock();
  cv_.notify_all();
  thread_.join();
}

template <concepts::Game Game>
void DataLoader<Game>::PrefetchThread::schedule_prefetch(DataFile* data_file) {
  mit::unique_lock lock(mutex_);
  file_ = data_file;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void DataLoader<Game>::PrefetchThread::loop() {
  while (!quitting_) {
    mit::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return quitting_ || file_ != nullptr; });
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
  for (int i = 0; i < num_prefetch_threads; ++i) {
    prefetch_threads_.push_back(new PrefetchThread(&thread_table_, i));
  }
  prefetch_loop_thread_ = mit::thread(&FileManager::prefetch_loop, this);
}

template <concepts::Game Game>
DataLoader<Game>::FileManager::~FileManager() {
  thread_table_.quit();
  exit_prefetch_loop();
  delete_all_files();

  for (PrefetchThread* thread : prefetch_threads_) {
    delete thread;
  }
  if (prefetch_loop_thread_.joinable()) {
    prefetch_loop_thread_.join();
  }
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::add_to_unload_queue(DataFile* file) {
  mit::unique_lock lock(mutex_);
  unload_queue_.push_back(file);
  RELEASE_ASSERT(active_file_count_ > 0, "FileManager::{}() bug", __func__);
  active_file_count_--;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::sort_work_units_and_prepare_files(work_unit_deque_t& work_units,
                                                                      generation_t* gen_range) {
  mit::unique_lock lock(mutex_);
  load_queue_.clear();
  unload_queue_.clear();
  active_file_count_ = 0;

  generation_t start_gen = -1;
  generation_t end_gen = -1;
  if (!work_units.empty()) {
    start_gen = work_units.back().file->gen();
    end_gen = work_units.front().file->gen();
    RELEASE_ASSERT(start_gen <= end_gen, "DataFileSet::{}() bug [start:{} end:{}]", __func__,
                   start_gen, end_gen);
  }

  // put work units with loaded files at the front
  std::sort(work_units.begin(), work_units.end(), [](const WorkUnit& a, const WorkUnit& b) {
    return a.file->is_loaded() > b.file->is_loaded();
  });

  int64_t expected_memory_usage = 0;
  for (const WorkUnit& unit : work_units) {
    DataFile* file = unit.file;
    if (file->is_loaded()) {
      active_file_count_++;
      expected_memory_usage += file->file_size();
    } else {
      load_queue_.push_back(file);
    }
  }

  if (start_gen >= 0) {
    trim(start_gen);
  }

  RELEASE_ASSERT(memory_usage_ == expected_memory_usage,
                 "DataFileSet::prepare_files() memory-usage-tracking-bug [{} != {}]", memory_usage_,
                 expected_memory_usage);

  lock.unlock();
  cv_.notify_all();

  gen_range[0] = start_gen;
  gen_range[1] = end_gen;
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::restore(int64_t n_total_rows, int n, generation_t* gens,
                                            int* row_counts, int64_t* file_sizes) {
  RELEASE_ASSERT(all_files_.empty(), "FileManager::init() bug");

  for (int i = 0; i < n; ++i) {
    append(gens[i], row_counts[i], file_sizes[i]);
  }
  n_total_rows_ = n_total_rows;
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::append(int end_gen, int num_rows, int64_t file_size) {
  auto filename = data_dir_ / std::format("gen-{}.data", end_gen);
  DataFile* data_file = new DataFile(filename.c_str(), end_gen, num_rows, file_size);

  mit::unique_lock lock(mutex_);
  n_total_rows_ += num_rows;
  all_files_.push_front(data_file);
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::reset_prefetch_loop() {
  // Resets the prefetch loop.
  //
  // We do this to ensure that when we call DataLoader::load(), we'll know for sure that we aren't
  // in the middle of prefetching any files. I haven't seen this happening, but I can't rule out
  // the possibility, so doing this seems like a good idea.
  mit::unique_lock lock(mutex_);
  quitting_ = true;
  lock.unlock();
  cv_.notify_all();

  if (prefetch_loop_thread_.joinable()) {
    prefetch_loop_thread_.join();
  }
  quitting_ = false;
  prefetch_loop_thread_ = mit::thread(&FileManager::prefetch_loop, this);
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
  RELEASE_ASSERT(!file->is_loaded(), "DataFileSet::prefetch_loop() bug at {}", __LINE__);

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

  RELEASE_ASSERT(file->file_size() > memory_budget_, "DataFileSet::prefetch_loop() bug at {}",
                 __LINE__);

  // our memory budget is insufficient to load this single file. In this case let's just violate
  // the memory budget and load the file anyway
  return kLoad;
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::prefetch_loop() {
  while (!quitting_) {
    mit::unique_lock lock(mutex_);
    Instruction instruction = kWait;
    cv_.wait(lock, [&] {
      instruction = get_next_instruction();
      return instruction != kWait;
    });

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
      throw util::Exception("DataFileSet::prefetch_loop() bug at {}", __LINE__);
    }
  }
}

template <concepts::Game Game>
void DataLoader<Game>::FileManager::exit_prefetch_loop() {
  mit::unique_lock lock(mutex_);
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
  thread_ = mit::thread(&WorkerThread::loop, this);
}

template <concepts::Game Game>
DataLoader<Game>::WorkerThread::~WorkerThread() {
  quit();
}

template <concepts::Game Game>
void DataLoader<Game>::WorkerThread::quit() {
  mit::unique_lock lock(mutex_);
  quitting_ = true;
  lock.unlock();
  cv_.notify_all();
  thread_.join();
}

template <concepts::Game Game>
void DataLoader<Game>::WorkerThread::schedule_work(const LoadInstructions& load_instructions,
                                                   const WorkUnit& unit) {
  mit::unique_lock lock(mutex_);
  load_instructions_ = &load_instructions;
  unit_ = unit;
  has_work_ = true;
  lock.unlock();
  cv_.notify_all();
}

template <concepts::Game Game>
void DataLoader<Game>::WorkerThread::loop() {
  while (!quitting_) {
    mit::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return quitting_ || has_work_; });
    if (quitting_) return;

    do_work();
    has_work_ = false;
    table_->mark_as_available(id_);
    file_manager_->add_to_unload_queue(unit_.file);
  }
}

template <concepts::Game Game>
void DataLoader<Game>::WorkerThread::do_work() {
  const local_index_vec_t& sample_indices = *unit_.sample_indices;

  int num_samples = sample_indices.size();
  if (num_samples == 0) return;

  DataFile* file = unit_.file;
  int output_index = unit_.output_index;

  const char* filename = file->filename();
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

  throw util::Exception("WorkerThread::do_work() bug at {} ({} != {}) num_games={}", __LINE__, s,
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
  thread_table_.quit();
  for (WorkerThread* thread : workers_) {
    delete thread;
  }
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

  thread_table_.wait_until_all_threads_available();
}

template <concepts::Game Game>
DataLoader<Game>::DataLoader(const Params& params)
    : params_(params),
      file_manager_(params.data_dir, params.memory_budget, params.num_prefetch_threads),
      work_manager_(&file_manager_, params.num_worker_threads) {}

template <concepts::Game Game>
void DataLoader<Game>::restore(int64_t n_total_rows, int n, generation_t* gens, int* row_counts,
                               int64_t* file_sizes) {
  file_manager_.restore(n_total_rows, n, gens, row_counts, file_sizes);
}

template <concepts::Game Game>
void DataLoader<Game>::add_gen(int gen, int num_rows, int64_t file_size) {
  file_manager_.append(gen, num_rows, file_size);
}

template <concepts::Game Game>
void DataLoader<Game>::load(int64_t window_start, int64_t window_end, int n_samples,
                            bool apply_symmetry, int n_targets, float* output_array,
                            int* target_indices_array, int* gen_range) {
  load_instructions_.init(apply_symmetry, n_targets, output_array, target_indices_array);
  sampling_manager_.sample(&work_units_, file_manager_.files_in_reverse_order(), window_start,
                           window_end, file_manager_.n_total_rows(), n_samples);
  file_manager_.sort_work_units_and_prepare_files(work_units_, gen_range);
  work_manager_.process(load_instructions_, work_units_);
  file_manager_.reset_prefetch_loop();
  shuffle_output(n_samples);
}

template <concepts::Game Game>
void DataLoader<Game>::shuffle_output(int n_samples) {
  float* f = load_instructions_.output_data_array;
  int row_size = load_instructions_.row_size;
  util::Random::chunked_shuffle(f, f + row_size * n_samples, row_size);
}

}  // namespace core
