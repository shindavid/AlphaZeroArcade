#include "util/EigenUtil.hpp"
#include "util/GTestUtil.hpp"
#include "util/Random.hpp"
#include "util/mit/exceptions.hpp"
#include "util/mit/mit.hpp"

#include <gtest/gtest.h>

#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

// A simple class that builds a 4x4 tensor in a multithreaded manner.
class TensorBuilder {
 public:
  using Tensor = eigen_util::FTensor<Eigen::Sizes<4, 4>>;

  TensorBuilder(bool use_main_thread) {
    for (int i = 0; i < 4; ++i) {
      if (i == 3 && use_main_thread) {
        fill_row(i);  // Fill the first row in the main thread
      } else {
        threads_.emplace_back([this, i]() { fill_row(i); });
      }
    }

    for (auto& thread : threads_) {
      thread.join();
    }
  }

  const Tensor& tensor() const { return tensor_; }

 private:
  void fill_row(int i) {
    for (int j = 0; j < 4; ++j) {
      mit::unique_lock lock(mutex_);
      tensor_(i, j) = next_value_++;
    }
  }
  std::vector<mit::thread> threads_;
  mit::mutex mutex_;

  Tensor tensor_;
  int next_value_ = 0;
};

class TensorBuildTest : public ::testing::Test {
 public:
  void run(bool use_main_thread, bool reseed) {
    std::vector<TensorBuilder::Tensor> tensors;
    int n = 5;
    int seed = 42;
    mit::seed(seed);

    for (int i = 0; i < n; ++i) {
      mit::reset();
      if (reseed) {
        mit::seed(seed);
      }
      TensorBuilder builder(use_main_thread);
      tensors.push_back(builder.tensor());
    }

    if (reseed) {
      // If reseeding, we expect all tensors to be equal
      for (int i = 1; i < n; ++i) {
        bool equal = eigen_util::equal(tensors[0], tensors[i]);
        EXPECT_TRUE(equal) << "tensors[0]: \n"
                           << tensors[0] << "\n"
                           << "tensors[" << i << "]: " << tensors[i];
      }
    } else {
      // If not reseeding, we expect variation
      for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
          bool equal = eigen_util::equal(tensors[i], tensors[j]);
          EXPECT_FALSE(equal) << "tensors[" << i << "]: \n"
                              << tensors[i] << "\n"
                              << "tensors[" << j << "]: " << tensors[j];
        }
      }
    }
  }
};

// These tests verify that TensorBuilder acts deterministically for a fixed seed, and that
// otherwise each run produces a different tensor.
//
// We run the tests with and without having the main thread perform business logic, in order to
// stress-test the mit::scheduler logic, as the main thread follows different code paths than other
// threads in the mit::scheduler implementation.
TEST_F(TensorBuildTest, test1) { run(true, true); }
TEST_F(TensorBuildTest, test2) { run(true, false); }
TEST_F(TensorBuildTest, test3) { run(false, true); }
TEST_F(TensorBuildTest, test4) { run(false, false); }

// A class that has a non-deterministic mutex deadlock bug
class MutexDeadlockBug {
 public:
  void run(bool use_main_thread) {
    mit::BugDetectGuard guard;  // Enable bug catching mode

    std::vector<mit::thread> threads;
    threads.emplace_back([this]() { func1(); });

    if (use_main_thread) {
      func2();  // Run func2 in the main thread
    } else {
      threads.emplace_back([this]() { func2(); });
    }

    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

 private:
  void func1() {
    mit::unique_lock lock1(m1_);
    mit::unique_lock lock2(m2_);
  }

  void func2() {
    mit::unique_lock lock2(m2_);
    mit::unique_lock lock1(m1_);
  }

  mit::mutex m1_, m2_;
};

class MutexDeadlockTest : public ::testing::Test {
 public:
  void run(bool use_main_thread, bool reseed) {
    mit::reset();
    mit::seed(42);

    MutexDeadlockBug bug;
    int throw_count = 0;
    int non_throw_count = 0;

    for (int i = 0; i < 100; ++i) {
      try {
        mit::reset();
        if (reseed) {
          mit::seed(42);
        }
        bug.run(use_main_thread);
        non_throw_count++;
      } catch (const mit::BugDetectedError&) {
        throw_count++;
      }
    }

    if (reseed) {
      EXPECT_EQ(throw_count * non_throw_count, 0)
        << "Expected either all threads to throw or none to throw, but got " << throw_count
        << " throws and " << non_throw_count << " non-throws.";
    } else {
      EXPECT_GT(throw_count, 10);
      EXPECT_GT(non_throw_count, 10);
    }
  }
};

// These tests verify that MutexDeadlockBug behaves deterministically for a fixed seed, and that
// otherwise it throws a BugDetectedError in some runs but not in others.
//
// We run the tests with and without having the main thread perform business logic, in order to
// stress-test the mit::scheduler logic, as the main thread follows different code paths than other
// threads in the mit::scheduler implementation.
TEST_F(MutexDeadlockTest, test1) { run(true, true); }
TEST_F(MutexDeadlockTest, test2) { run(true, false); }
TEST_F(MutexDeadlockTest, test3) { run(false, true); }
TEST_F(MutexDeadlockTest, test4) { run(false, false); }

// A class that non-deterministically fails to issue a condition variable notify
class ConditionVariableNotifyBug {
 public:
  void run(bool use_main_thread) {
    mit::BugDetectGuard guard;  // Enable bug catching mode
    notified_ = false;
    std::vector<mit::thread> threads;
    threads.emplace_back([this]() { func1(); });
    if (use_main_thread) {
      func2();  // Run func2 in the main thread
    } else {
      threads.emplace_back([this]() { func2(); });
    }
    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

 private:
  void func1() {
    mit::unique_lock lock(mutex_);
    cv_.wait(lock, [this]() { return notified_; });
  }

  void func2() {
    mit::unique_lock lock(mutex_);
    notified_ = true;
    if (util::Random::uniform_real(0.0, 1.0) < 0.5) {
      // Randomly fail to notify
      return;
    }
    cv_.notify_one();
  }

  mit::mutex mutex_;
  mit::condition_variable cv_;
  bool notified_ = false;
};

class ConditionVariableNotifyTest : public ::testing::Test {
 public:
  void run(bool use_main_thread, bool reseed) {
    mit::reset();
    util::Random::set_seed(42);
    mit::seed(42);

    ConditionVariableNotifyBug bug;
    int throw_count = 0;
    int non_throw_count = 0;

    for (int i = 0; i < 100; ++i) {
      try {
        mit::reset();
        if (reseed) {
          util::Random::set_seed(42);
          mit::seed(42);
        }
        bug.run(use_main_thread);
        non_throw_count++;
      } catch (const mit::BugDetectedError&) {
        throw_count++;
      }
    }

    if (reseed) {
      EXPECT_EQ(throw_count * non_throw_count, 0)
        << "Expected either all threads to throw or none to throw, but got " << throw_count
        << " throws and " << non_throw_count << " non-throws.";
    } else {
      EXPECT_GT(throw_count, 10);
      EXPECT_GT(non_throw_count, 10);
    }
  }
};

// These tests verify that ConditionVariableNotifyBug behaves deterministically for a fixed seed,
// and that otherwise it throws a BugDetectedError in some runs but not in others.
//
// We run the tests with and without having the main thread perform business logic, in order to
// stress-test the mit::scheduler logic, as the main thread follows different code paths than other
// threads in the mit::scheduler implementation.
TEST_F(ConditionVariableNotifyTest, test1) { run(true, true); }
TEST_F(ConditionVariableNotifyTest, test2) { run(true, false); }
TEST_F(ConditionVariableNotifyTest, test3) { run(false, true); }
TEST_F(ConditionVariableNotifyTest, test4) { run(false, false); }

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
