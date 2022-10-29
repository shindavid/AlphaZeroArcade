#include <iostream>

#include <connect4/C4GameLogic.hpp>
#include <util/BitSet.hpp>

#include <Eigen/Core>
#include <torch/torch.h>

int main() {
  using bitset_t = util::BitSet<8>;
  bitset_t bits;
  bits[1] = true;
  bits[4] = true;
  for (auto it : bits) {
    std::cout << it << std::endl;
  }
  torch::Tensor tensor = torch::rand({2, 3});
  torch::Tensor tensor2 = torch::rand({5, 3});
  torch::save({tensor, tensor2}, "test.pt");

  using namespace Eigen;
  Matrix3f m3;
  m3 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  Matrix4f m4 = Matrix4f::Identity();
  Vector4i v4(1, 2, 3, 4);

  std::cout << "m3\n" << m3 << "\nm4:\n"
    << m4 << "\nv4:\n" << v4 << std::endl;

  c4::GameState state;
  return 0;
}
