#include <sstream>

#include <common/AbstractSymmetryTransform.hpp>
#include <common/DerivedTypes.hpp>
#include <othello/Constants.hpp>
#include <othello/GameState.hpp>
#include <othello/Tensorizor.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

using SymmetryTransform = common::AbstractSymmetryTransform<othello::GameState, othello::Tensorizor>;
using GameStateTypes = common::GameStateTypes<othello::GameState>;
using TensorizorTypes = common::TensorizorTypes<othello::Tensorizor>;
using PolicyEigenTensor = GameStateTypes::PolicyEigenTensor;
using InputEigenTensor = TensorizorTypes::InputEigenTensor;

int test_symmetry_input(
    SymmetryTransform& transform,
    const std::string& expected_repr0,
    const std::string& expected_repr1)
{
  InputEigenTensor input;
  for (int i = 0; i < eigen_util::total_size_v<eigen_util::extract_sizes_t<InputEigenTensor>>; ++i) {
    input.data()[i] = i;
  }

  transform.transform_input(input);

  using SliceShape = Eigen::Sizes<8, 8>;
  using InputEigenTensorSlice = Eigen::TensorFixedSize<float, SliceShape, Eigen::RowMajor>;

  for (int slice = 0; slice < 2; ++slice) {
    const std::string& expected_repr = slice == 0 ? expected_repr0 : expected_repr1;

    InputEigenTensorSlice input_slice;
    eigen_util::packed_fixed_tensor_cp(input_slice, eigen_util::slice<SliceShape>(input, slice));
    std::ostringstream ss;
    ss << input_slice;
    std::string repr = ss.str();

    if (repr != expected_repr) {
      printf("%s() %s failure!\n", __func__, util::get_typename(transform).c_str());
      std::cout << "Expected:" << std::endl;
      std::cout << expected_repr << std::endl;
      std::cout << "But got:" << std::endl;
      std::cout << repr << std::endl;
      return 1;
    }
  }
  return 0;
}

int test_symmetry_policy(SymmetryTransform& transform, const std::string expected_repr) {
  PolicyEigenTensor policy;

  for (int i = 0; i < eigen_util::total_size_v<eigen_util::extract_sizes_t<PolicyEigenTensor>>; ++i) {
    policy.data()[i] = i;
  }

  transform.transform_policy(policy);
  std::ostringstream ss;
  ss << policy;
  std::string repr = ss.str();

  if (repr != expected_repr) {
    printf("%s() %s failure!\n", __func__, util::get_typename(transform).c_str());
    std::cout << "Expected:" << std::endl;
    std::cout << expected_repr << std::endl;
    std::cout << "But got:" << std::endl;
    std::cout << repr << std::endl;
    return 1;
  }
  return 0;
}

int test_identity() {
  othello::Tensorizor::IdentityTransform transform;

  std::string expected_input_slice0 = " 0  1  2  3  4  5  6  7\n"
                                      " 8  9 10 11 12 13 14 15\n"
                                      "16 17 18 19 20 21 22 23\n"
                                      "24 25 26 27 28 29 30 31\n"
                                      "32 33 34 35 36 37 38 39\n"
                                      "40 41 42 43 44 45 46 47\n"
                                      "48 49 50 51 52 53 54 55\n"
                                      "56 57 58 59 60 61 62 63";

  std::string expected_input_slice1 = " 64  65  66  67  68  69  70  71\n"
                                      " 72  73  74  75  76  77  78  79\n"
                                      " 80  81  82  83  84  85  86  87\n"
                                      " 88  89  90  91  92  93  94  95\n"
                                      " 96  97  98  99 100 101 102 103\n"
                                      "104 105 106 107 108 109 110 111\n"
                                      "112 113 114 115 116 117 118 119\n"
                                      "120 121 122 123 124 125 126 127";

  std::string expected_policy = " 0  1  2  3  4  5  6  7\n"
                                " 8  9 10 11 12 13 14 15\n"
                                "16 17 18 19 20 21 22 23\n"
                                "24 25 26 27 28 29 30 31\n"
                                "32 33 34 35 36 37 38 39\n"
                                "40 41 42 43 44 45 46 47\n"
                                "48 49 50 51 52 53 54 55\n"
                                "56 57 58 59 60 61 62 63";

  int fail_count = 0;
  fail_count += test_symmetry_input(transform, expected_input_slice0, expected_input_slice1);
  fail_count += test_symmetry_policy(transform, expected_policy);
  return fail_count;
}

int test_rotation90() {
  othello::Tensorizor::Rotation90Transform transform;

  std::string expected_input_slice0 = "56 48 40 32 24 16  8  0\n"
                                      "57 49 41 33 25 17  9  1\n"
                                      "58 50 42 34 26 18 10  2\n"
                                      "59 51 43 35 27 19 11  3\n"
                                      "60 52 44 36 28 20 12  4\n"
                                      "61 53 45 37 29 21 13  5\n"
                                      "62 54 46 38 30 22 14  6\n"
                                      "63 55 47 39 31 23 15  7";

  std::string expected_input_slice1 = "120 112 104  96  88  80  72  64\n"
                                      "121 113 105  97  89  81  73  65\n"
                                      "122 114 106  98  90  82  74  66\n"
                                      "123 115 107  99  91  83  75  67\n"
                                      "124 116 108 100  92  84  76  68\n"
                                      "125 117 109 101  93  85  77  69\n"
                                      "126 118 110 102  94  86  78  70\n"
                                      "127 119 111 103  95  87  79  71";

  std::string expected_policy = " 7 15 23 31 39 47 55 63\n"
                                " 6 14 22 30 38 46 54 62\n"
                                " 5 13 21 29 37 45 53 61\n"
                                " 4 12 20 28 36 44 52 60\n"
                                " 3 11 19 27 35 43 51 59\n"
                                " 2 10 18 26 34 42 50 58\n"
                                " 1  9 17 25 33 41 49 57\n"
                                " 0  8 16 24 32 40 48 56";

  int fail_count = 0;
  fail_count += test_symmetry_input(transform, expected_input_slice0, expected_input_slice1);
  fail_count += test_symmetry_policy(transform, expected_policy);
  return fail_count;
}

int test_symmetries() {
  int fail_count = 0;
  fail_count += test_identity();

//  othello::Tensorizor::Rotation90Transform rot90;
//  othello::Tensorizor::Rotation180Transform rot180;
//  othello::Tensorizor::Rotation270Transform rot270;
//  othello::Tensorizor::ReflectionOverHorizontalTransform ref_horiz;
//  othello::Tensorizor::ReflectionOverHorizontalWithRotation90Transform ref_horiz_rot90;
//  othello::Tensorizor::ReflectionOverHorizontalWithRotation180Transform ref_horiz_rot180;
//  othello::Tensorizor::ReflectionOverHorizontalWithRotation270Transform ref_horiz_rot270;

  return fail_count;
}

int main() {
  int fail_count = 0;
  fail_count += test_symmetries();

  if (fail_count) {
    printf("Failed %d test%s!\n", fail_count, fail_count > 1 ? "s" : "");
  } else {
    printf("All tests passed!\n");
  }
  return 0;
}