/*
 *
 */
#include <torch/torch.h>

#include <vector>

#include <common/NeuralNet.hpp>
#include <util/CudaUtil.hpp>
#include <util/RepoUtil.hpp>

void experiment(const char* filename, int batch_size) {
  torch::jit::script::Module module(torch::jit::load(filename));
  module.to(torch::kCUDA);
  torch::Tensor input = torch::zeros({batch_size, 2, 7, 6});
  torch::Tensor output = torch::zeros({batch_size, 7});

  std::vector<torch::jit::IValue> input_vec;

  for (int i = 0; i < 10000; ++i) {
    if (i < 10 || i % 100 == 0) {
      printf("%6d ", i);
      std::cout << output.data_ptr() << " " << output.sizes() << " ";
      cuda::dump_memory_info();
    }
    torch::Tensor gpu_input = input.clone().to(torch::kCUDA);
    input_vec.push_back(gpu_input);
    auto gpu_output = module.forward(input_vec).toTuple()->elements()[0].toTensor();
    output.copy_(gpu_output.detach());
    input_vec.clear();
  }
}

int main(int ac, char* av[]) {
  int batch_size = atoi(av[1]);
  printf("batch_size: %d\n", batch_size);
  auto path = util::Repo::root() / "c4_model.pt";
  experiment(path.c_str(), batch_size);
}