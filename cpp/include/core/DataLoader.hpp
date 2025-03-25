#pragma once

#include <core/concepts/Game.hpp>
#include <core/GameLog.hpp>

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/*
 * core::DataLoader<Game> is a class that is used on the python side via FFI to generate
 * minibatches of training data. We use it instead of pytorch's DataLoader.
 *
 **************
 * Background *
 **************
 *
 * The self-play data is stored in game log files: one per generation of data. In a sqlite3
 * database, we maintain information on how many rows are in each file. If we imagine concatenating
 * all the files together, we can imagine a single array of rows that houses all the data. Let us
 * call this array M (for "master array").
 *
 * The python side constructs a DataLoader object via an FFI interface, and passes it the following
 * parameters:
 *
 *    - s: num samples (typically equals n_minibatches * minibatch_size)
 *    - w: window size
 *
 * Then core::DataLoader<Game> samples s rows from M[-w:].
 *
 *************
 * Mechanics *
 *************
 *
 * core::DataLoader<Game> starts by sampling m*b indices, withouth replacement, from M[c:]. It then
 * sorts these indices, grouping them by file. Each file can then be read in a single pass.
 *
 * The work of reading each file is done by a worker thread. There will be several worker threads,
 * coming from a worker pool. This allows us to read multiple files in parallel. Additionally, we
 * will have a prefetch pool, consisting of prefetch threads. These threads open up the next file in
 * the list, storing the file bytes in memory. This way, the worker threads can immediately start
 * reading from the next file, without waiting for the filesystem.
 */

namespace core {

template <concepts::Game Game>
class DataLoader {
 public:
  using GameReadLog = core::GameReadLog<Game>;
  using thread_id_t = int32_t;
  using generation_t = int32_t;
  using target_index_t = int32_t;
  using global_index_t = int64_t;      // index within master list M
  using rev_global_index_t = int64_t;  // index within master list M, from the end
  using local_index_t = int32_t;       // index within a single file
  using local_index_vec_t = std::vector<local_index_t>;
  using sampling_plan_t = std::map<generation_t, local_index_vec_t>;
  using rev_global_index_vec_t = std::vector<rev_global_index_t>;

  // Used to pass parameters to the DataLoader constructor
  struct Params {
    std::string data_dir;      // directory containing the game log files
    int64_t memory_budget;     // memory budget for data files, in bytes
    int num_worker_threads;    // number of worker threads
    int num_prefetch_threads;  // number of prefetch threads
  };

  class DataFile {
   public:
    DataFile(const char* filename, int gen, int num_rows, int64_t file_size);
    ~DataFile();

    const char* filename() const { return filename_.c_str(); }
    generation_t gen() const { return gen_; }
    const int num_rows() const { return num_rows_; }
    int64_t file_size() const { return file_size_; }

    // Loads the file's contents into memory.
    void load();

    // Unloads the file if it's currently loaded. Returns the memory freed.
    int64_t unload();

    // Blocks until the file is loaded, and then returns a pointer to the file's contents.
    const char* buffer() const;

    bool is_loaded() const { return buffer_ != nullptr; }

   private:
    std::string filename_;
    const generation_t gen_;
    const int num_rows_;
    const int64_t file_size_;

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    char* buffer_ = nullptr;
    bool locked_ = false;
  };
  using file_deque_t = std::deque<DataFile*>;

  struct WorkUnit {
    DataFile* file;
    const local_index_vec_t* sample_indices;
    int output_index;
  };
  using work_unit_deque_t = std::deque<WorkUnit>;

  /*
   * LoadInstructions specifies what tensors to extract from the game data and where to write them.
   */
  struct LoadInstructions {
    void init(bool apply_sym, int n_targets, float* output_array, int* target_indices_array);

    bool apply_symmetry;
    float* output_data_array;
    std::vector<int> target_indices;
    int row_size;
  };

  /*
   * The SamplingManager is responsible for sampling from the master list M, and organizing the
   * sampled indices into WorkUnit's.
   */
  class SamplingManager {
   public:
    ~SamplingManager();

    // Sample from M[window_start:window_end]. Writes output into work_units. A work unit gets
    // created for every gen that intersects M[-window_size:], even if no samples are taken from
    // that gen.
    //
    // We expect files to be in reverse order by generation.
    void sample(work_unit_deque_t* work_units, const file_deque_t& files, int64_t window_start,
                int64_t window_end, int64_t n_total_rows, int n_samples);

   private:
    local_index_vec_t* get_vec();
    void reset_vec_pools();

    using vec_pool_t = std::vector<local_index_vec_t*>;

    // We recycle local_index_vec_t objects to minimize memory allocation overhead
    vec_pool_t vec_pool_used_;
    vec_pool_t vec_pool_unused_;
    rev_global_index_vec_t sampled_indices_;  // temp storage to support sample()
  };

  /*
   * A ThreadTable is used by a manager to keep track of its threads.
   */
  class ThreadTable {
   public:
    ThreadTable(int n_threads);
    void mark_as_available(thread_id_t id);

    // Blocks until a thread is available. Then marks it as unavailable and returns its id.
    // If quitting_ is true, returns immediately and returns -1.
    thread_id_t allocate_thread();

    // Blocks until all threads are available, or until quitting_ is true.
    void wait_until_all_threads_available();

    // Force the blocking methods to return immediately
    void quit();

   private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::vector<thread_id_t> available_thread_ids_;
    const int n_threads_;
    bool quitting_ = false;
  };

  /*
   * A PrefetchThread is responsible for loading files into memory. It is given work to do by the
   * FileManager.
   */
  class PrefetchThread {
   public:
    PrefetchThread(ThreadTable* table, thread_id_t id);
    ~PrefetchThread();

    void quit();
    void schedule_prefetch(DataFile* data_file);

   private:
    void loop();

    ThreadTable* const table_;
    const thread_id_t id_;

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::thread thread_;
    DataFile* file_ = nullptr;
    bool quitting_ = false;
  };

  /*
   * The FileManager is responsible for maintaining a list of DataFile's, and for managing the
   * loading and unloading of their contents. This entails considerations like pre-fetching data
   * from disk into memory to minimize the time spent by WorkerThread's waiting for I/O, while also
   * ensuring that the memory budget is not exceeded.
   */
  class FileManager {
   public:
    FileManager(const boost::filesystem::path& data_dir, int64_t memory_budget,
                int num_prefetch_threads);
    ~FileManager();

    // Launches background prefetching of files. Writes the first and last gen of the sampled rows
    // into gen_range.
    void prepare_files(const work_unit_deque_t& work_units, generation_t* gen_range);

    // Called at startup to restore an existing run
    //
    // Each of the arrays will be of length n. The k'th generation of self-play data will be for
    // generation gens[k], with row_counts[k] rows and file_sizes[k] bytes.
    void restore(int64_t n_total_rows, int n, generation_t* gens, int* row_counts,
                 int64_t* file_sizes);

    // Add a new generation of data to the manager
    void append(generation_t gen, int num_rows, int64_t file_size);

    // Returns the list of files in reverse order by generation
    const file_deque_t& files_in_reverse_order() const { return all_files_; }

    void decrement_active_file_count();

    int64_t n_total_rows() const { return n_total_rows_; }

   private:
    enum Instruction : int8_t { kUnload, kLoad, kWait, kQuit };

    // Unload data files that are no longer needed
    void trim(generation_t start_gen);

    Instruction get_next_instruction() const;
    void prefetch_loop();
    void exit_prefetch_loop();
    void delete_all_files();

    const boost::filesystem::path data_dir_;
    const int64_t memory_budget_;

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::thread prefetch_loop_thread_;
    bool quitting_ = false;

    std::vector<PrefetchThread*> prefetch_threads_;
    ThreadTable thread_table_;

    file_deque_t load_queue_;
    file_deque_t unload_queue_;
    int active_file_count_ = 0;

    int64_t n_total_rows_ = 0;
    file_deque_t all_files_;  // stored in reverse order by generation
    int64_t memory_usage_ = 0;  // sum(file->memory_usage() for all files in all_files_)
  };

  /*
   * A WorkerThread is responsible for doing the main work of processing a WorkUnit. It is given
   * work to do by the WorkManager.
   */
  class WorkerThread {
   public:
    WorkerThread(FileManager* file_manager, ThreadTable* table, thread_id_t id);
    ~WorkerThread();

    void quit();
    void schedule_work(const LoadInstructions& load_instructions, const WorkUnit& unit);

   private:
    void loop();
    void do_work();

    FileManager* const file_manager_;
    ThreadTable* const table_;
    const thread_id_t id_;

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::thread thread_;
    const LoadInstructions* load_instructions_;
    WorkUnit unit_;
    bool quitting_ = false;
    bool has_work_ = false;
  };

  /*
   * The WorkManager is responsible for managing a collection of WorkUnit's, and distributing them
   * to WorkerThread's for processing.
   */
  class WorkManager {
   public:
    WorkManager(FileManager* file_manager, int num_threads);
    ~WorkManager();

    // Processes the list of work units. Pops work units from the front, assigns to a worker, until
    // the list is empty. Returns when all work units have been processed.
    void process(const LoadInstructions& load_instructions, work_unit_deque_t& work_units);

   private:
    std::vector<WorkerThread*> workers_;
    ThreadTable thread_table_;
  };

  DataLoader(const Params&);

  void restore(int64_t n_total_rows, int n, generation_t* gens, int* row_counts,
               int64_t* file_sizes);

  void add_gen(int gen, int num_rows, int64_t file_size);

  /*
   * Samples n_samples rows from M[window_size:window_end]. Converts the rows into tensors, which
   * are written into output_array. Writes the first and last gen of the sampled rows into
   * gen_range.
   *
   * On the python side, output_array is sized to fit all the tensors that will be generated. Each
   * row is expected to be written as a concatenated (input, targets..., masks...) row. The python
   * will take care of subsequently splitting the row into its constituent parts.
   *
   * NOTE: if the tensors are very large, this might not all fit in memory. In that case, it is up
   * to the caller to call load() multiple times, passing in values for n_minibatches that sum up to
   * the desired total.
   */
  void load(int64_t window_start, int64_t window_end, int n_samples, bool apply_symmetry,
            int n_targets, float* output_array, int* target_indices_array, int* gen_range);

 private:
  void shuffle_output(int n_samples);

  const Params params_;
  LoadInstructions load_instructions_;

  SamplingManager sampling_manager_;
  FileManager file_manager_;
  WorkManager work_manager_;
  work_unit_deque_t work_units_;
};

}  // namespace core

#include <inline/core/DataLoader.inl>
