/*
 * Usage:
 *
 * ./dshin_scratch c4_model.pt 144 100  # batch-size=144, loop-size=100
 */
#include <torch/torch.h>

#include <chrono>
#include <vector>

#include <core/NeuralNet.hpp>
#include <util/CudaUtil.hpp>
#include <util/RepoUtil.hpp>

void experiment(const char* filename, int batch_size, int loop_size) {
  using clock_t = std::chrono::steady_clock;
  using time_point_t = std::chrono::time_point<clock_t>;

  std::cout << "Loading model..." << std::endl;
  torch::jit::script::Module module(torch::jit::load(filename));
  module.to(torch::kCUDA);
  torch::Tensor input = torch::zeros({batch_size, 2, 7, 6});
  torch::Tensor output = torch::zeros({batch_size, 7});

  std::vector<torch::jit::IValue> input_vec;

  // warm-up
  std::cout << "Warming up GPU..." << std::endl;
  for (int i = 0; i < 5; ++i) {
    torch::Tensor gpu_input = input.clone().to(torch::kCUDA);
    input_vec.push_back(gpu_input);
    auto gpu_output = module.forward(input_vec).toTuple()->elements()[0].toTensor();
    output.copy_(gpu_output.detach());
    input_vec.clear();
  }

  std::cout << "Performing experiment..." << std::endl;
  time_point_t t1 = clock_t::now();
  for (int i = 0; i < loop_size; ++i) {
    torch::Tensor gpu_input = input.clone().to(torch::kCUDA);
    input_vec.push_back(gpu_input);
    auto gpu_output = module.forward(input_vec).toTuple()->elements()[0].toTensor();
    output.copy_(gpu_output.detach());
    input_vec.clear();
  }
  time_point_t t2 = clock_t::now();
  std::chrono::nanoseconds duration = t2 - t1;
  int64_t ns = duration.count();
  double us_per_eval = ns * 1e-3 / (loop_size * batch_size);
  double us_per_batch = ns * 1e-3 / loop_size;

  printf("model: %s\n", filename);
  printf("batch_size: %d\n", batch_size);
  printf("loop_size: %d\n", loop_size);
  printf("us_per_batch: %.3f\n", us_per_batch);
  printf("us_per_eval: %.3f\n", us_per_eval);
}

int main(int ac, char* av[]) {
  const char* filename = av[1];
  int batch_size = atoi(av[2]);
  int loop_size = atoi(av[3]);
  experiment(filename, batch_size, loop_size);
}
