#include <gtest/gtest.h>
#include <util/EigenUtil.hpp>
#include <util/GTestUtil.hpp>
#include <util/mit/exceptions.hpp>
#include <util/mit/mit.hpp>

#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

// A simple class that builds a 4x4 tensor in a multithreaded manner.
class TensorBuilder {
 public:
  using Tensor = eigen_util::FTensor<Eigen::Sizes<4, 4>>;

  TensorBuilder() {
    for (int i = 0; i < 4; ++i) {
      threads_.emplace_back([this, i]() {
        for (int j = 0; j < 4; ++j) {
          mit::unique_lock lock(mutex_);
          tensor_(i, j) = next_value_++;
        }
      });
    }

    for (auto& thread : threads_) {
      thread.join();
    }
  }

  const Tensor& tensor() const { return tensor_; }

 private:
  std::vector<mit::thread> threads_;
  mit::mutex mutex_;

  Tensor tensor_;
  int next_value_ = 0;
};

// Validate that TensorBuilder is deterministic for a fixed seed
TEST(mit, deterministic_threading) {
  std::vector<TensorBuilder::Tensor> tensors;
  int seed = 42;

  for (int i = 0; i < 5; ++i) {
    mit::reset();
    mit::seed(seed);
    TensorBuilder builder;
    tensors.push_back(builder.tensor());
  }

  for (int i = 1; i < (int)tensors.size(); ++i) {
    bool equal = eigen_util::equal(tensors[0], tensors[i]);
    EXPECT_TRUE(equal) << "tensors[0]: \n" << tensors[0] << "\n"
                       << "tensors[" << i << "]: " << tensors[i];
  }
}

// Validate that TensorBuilder produces different tensors with different seeds
TEST(mit, seed_matters) {
  mit::reset();
  mit::seed(42);
  TensorBuilder builder1;
  auto tensor1 = builder1.tensor();

  mit::reset();
  mit::seed(43);
  TensorBuilder builder2;
  auto tensor2 = builder2.tensor();

  bool equal = eigen_util::equal(tensor1, tensor2);
  EXPECT_FALSE(equal) << "tensor1: \n" << tensor1 << "\n"
                      << "tensor2: " << tensor2;
}

// A class that has a non-deterministic mutex deadlock bug
class DeadlockBug {
 public:
  void run() {
    mit::BugDetectGuard guard;  // Enable bug catching mode

    mit::mutex m1, m2;

    mit::thread t1([&]() {
      mit::unique_lock lock1(m1);
      mit::unique_lock lock2(m2);
    });

    mit::thread t2([&]() {
      mit::unique_lock lock2(m2);
      mit::unique_lock lock1(m1);
    });

    if (t1.joinable()) t1.join();
    if (t2.joinable()) t2.join();
  }
};

TEST(mit, deadlock_bug) {
  mit::reset();
  mit::seed(42);

  DeadlockBug bug;
  int throw_count = 0;
  int non_throw_count = 0;

  for (int i = 0; i < 100; ++i) {
    try {
      mit::reset();
      bug.run();
      non_throw_count++;
    } catch (const mit::BugDetectedError&) {
      throw_count++;
    }
  }

  EXPECT_GT(throw_count, 0);
  EXPECT_GT(non_throw_count, 0);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
