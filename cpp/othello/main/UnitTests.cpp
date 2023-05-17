#include <sstream>

#include <common/DerivedTypes.hpp>
#include <othello/GameState.hpp>
#include <othello/Tensorizor.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

/*
 * Tests othello symmetry classes.
 *
 * Note that the policy transforms do not act the way you might expect on the central 4 squares of the board. See
 * comments in cpp/othello/Tensorizor.hpp
 */

using GameStateTypes = common::GameStateTypes<othello::GameState>;
using TensorizorTypes = common::TensorizorTypes<othello::Tensorizor>;
using PolicyTensor = GameStateTypes::PolicyTensor;
using InputTensor = Eigen::TensorFixedSize<int, TensorizorTypes::InputShape, Eigen::RowMajor>;
using InputScalar = InputTensor::Scalar;

int global_pass_count = 0;
int global_fail_count = 0;

template<typename TransformT>
void test_symmetry_input(
    TransformT& transform,
    const std::string& expected_repr0,
    const std::string& expected_repr1)
{
  InputTensor input;
  for (int i = 0; i < eigen_util::extract_shape_t<InputTensor>::total_size; ++i) {
    input.data()[i] = i;
  }

  try {
    transform.transform_input(input);
  } catch (...) {
    printf("%s() %s failure (caught exception)!\n", __func__, util::get_typename(transform).c_str());
    global_fail_count++;
    return;
  }

  using SliceShape = Eigen::Sizes<8, 8>;
  using InputTensorSlice = Eigen::TensorFixedSize<InputScalar, SliceShape, Eigen::RowMajor>;

  for (int slice = 0; slice < 2; ++slice) {
    const std::string &expected_repr = slice == 0 ? expected_repr0 : expected_repr1;

    InputTensorSlice input_slice;
    eigen_util::packed_fixed_tensor_cp(input_slice, eigen_util::slice(input, slice));
    std::ostringstream ss;
    ss << input_slice;
    std::string repr = ss.str();

    if (repr != expected_repr) {
      printf("%s() %s failure!\n", __func__, util::get_typename(transform).c_str());
      printf("Expected (slice %d):\n", slice);
      std::cout << expected_repr << std::endl;
      std::cout << "But got:" << std::endl;
      std::cout << repr << std::endl;
      global_fail_count++;
      return;
    }
  }
  printf("%s() %s success!\n", __func__, util::get_typename(transform).c_str());
  global_pass_count++;
}

template<typename TransformT>
void test_symmetry_policy(TransformT& transform, const std::string& expected_repr) {
  PolicyTensor policy;

  for (int i = 0; i < eigen_util::extract_shape_t<PolicyTensor>::total_size; ++i) {
    policy.data()[i] = i;
  }

  try {
    transform.transform_policy(policy);
  } catch (...) {
    printf("%s() %s failure (caught exception)!\n", __func__, util::get_typename(transform).c_str());
    global_fail_count++;
    return;
  }
  std::ostringstream ss;
  ss << policy;
  std::string repr = ss.str();

  if (repr != expected_repr) {
    printf("%s() %s failure!\n", __func__, util::get_typename(transform).c_str());
    std::cout << "Expected:" << std::endl;
    std::cout << expected_repr << std::endl;
    std::cout << "But got:" << std::endl;
    std::cout << repr << std::endl;
    global_fail_count++;
    return;
  } else {
    printf("%s() %s success!\n", __func__, util::get_typename(transform).c_str());
    global_pass_count++;
    return;
  }
}

void test_identity() {
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

  test_symmetry_input(transform, expected_input_slice0, expected_input_slice1);
  test_symmetry_policy(transform, expected_policy);
}

void test_rotation90() {
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
                                " 4 12 20 27 28 44 52 60\n"
                                " 3 11 19 35 36 43 51 59\n"
                                " 2 10 18 26 34 42 50 58\n"
                                " 1  9 17 25 33 41 49 57\n"
                                " 0  8 16 24 32 40 48 56";

  test_symmetry_input(transform, expected_input_slice0, expected_input_slice1);
  test_symmetry_policy(transform, expected_policy);
}

void test_rotation180() {
  othello::Tensorizor::Rotation180Transform transform;

  std::string expected_input_slice0 = "63 62 61 60 59 58 57 56\n"
                                      "55 54 53 52 51 50 49 48\n"
                                      "47 46 45 44 43 42 41 40\n"
                                      "39 38 37 36 35 34 33 32\n"
                                      "31 30 29 28 27 26 25 24\n"
                                      "23 22 21 20 19 18 17 16\n"
                                      "15 14 13 12 11 10  9  8\n"
                                      " 7  6  5  4  3  2  1  0";

  std::string expected_input_slice1 = "127 126 125 124 123 122 121 120\n"
                                      "119 118 117 116 115 114 113 112\n"
                                      "111 110 109 108 107 106 105 104\n"
                                      "103 102 101 100  99  98  97  96\n"
                                      " 95  94  93  92  91  90  89  88\n"
                                      " 87  86  85  84  83  82  81  80\n"
                                      " 79  78  77  76  75  74  73  72\n"
                                      " 71  70  69  68  67  66  65  64";

  std::string expected_policy = "63 62 61 60 59 58 57 56\n"
                                "55 54 53 52 51 50 49 48\n"
                                "47 46 45 44 43 42 41 40\n"
                                "39 38 37 27 28 34 33 32\n"
                                "31 30 29 35 36 26 25 24\n"
                                "23 22 21 20 19 18 17 16\n"
                                "15 14 13 12 11 10  9  8\n"
                                " 7  6  5  4  3  2  1  0";

  test_symmetry_input(transform, expected_input_slice0, expected_input_slice1);
  test_symmetry_policy(transform, expected_policy);
}

void test_rotation270() {
  othello::Tensorizor::Rotation270Transform transform;

  std::string expected_input_slice0 = " 7 15 23 31 39 47 55 63\n"
                                      " 6 14 22 30 38 46 54 62\n"
                                      " 5 13 21 29 37 45 53 61\n"
                                      " 4 12 20 28 36 44 52 60\n"
                                      " 3 11 19 27 35 43 51 59\n"
                                      " 2 10 18 26 34 42 50 58\n"
                                      " 1  9 17 25 33 41 49 57\n"
                                      " 0  8 16 24 32 40 48 56";

  std::string expected_input_slice1 = " 71  79  87  95 103 111 119 127\n"
                                      " 70  78  86  94 102 110 118 126\n"
                                      " 69  77  85  93 101 109 117 125\n"
                                      " 68  76  84  92 100 108 116 124\n"
                                      " 67  75  83  91  99 107 115 123\n"
                                      " 66  74  82  90  98 106 114 122\n"
                                      " 65  73  81  89  97 105 113 121\n"
                                      " 64  72  80  88  96 104 112 120";

  std::string expected_policy = "56 48 40 32 24 16  8  0\n"
                                "57 49 41 33 25 17  9  1\n"
                                "58 50 42 34 26 18 10  2\n"
                                "59 51 43 27 28 19 11  3\n"
                                "60 52 44 35 36 20 12  4\n"
                                "61 53 45 37 29 21 13  5\n"
                                "62 54 46 38 30 22 14  6\n"
                                "63 55 47 39 31 23 15  7";

  test_symmetry_input(transform, expected_input_slice0, expected_input_slice1);
  test_symmetry_policy(transform, expected_policy);
}

void test_reflection_over_horizontal() {
  othello::Tensorizor::ReflectionOverHorizontalTransform transform;

  std::string expected_input_slice0 = "56 57 58 59 60 61 62 63\n"
                                      "48 49 50 51 52 53 54 55\n"
                                      "40 41 42 43 44 45 46 47\n"
                                      "32 33 34 35 36 37 38 39\n"
                                      "24 25 26 27 28 29 30 31\n"
                                      "16 17 18 19 20 21 22 23\n"
                                      " 8  9 10 11 12 13 14 15\n"
                                      " 0  1  2  3  4  5  6  7";

  std::string expected_input_slice1 = "120 121 122 123 124 125 126 127\n"
                                      "112 113 114 115 116 117 118 119\n"
                                      "104 105 106 107 108 109 110 111\n"
                                      " 96  97  98  99 100 101 102 103\n"
                                      " 88  89  90  91  92  93  94  95\n"
                                      " 80  81  82  83  84  85  86  87\n"
                                      " 72  73  74  75  76  77  78  79\n"
                                      " 64  65  66  67  68  69  70  71";

  std::string expected_policy = "56 57 58 59 60 61 62 63\n"
                                "48 49 50 51 52 53 54 55\n"
                                "40 41 42 43 44 45 46 47\n"
                                "32 33 34 27 28 37 38 39\n"
                                "24 25 26 35 36 29 30 31\n"
                                "16 17 18 19 20 21 22 23\n"
                                " 8  9 10 11 12 13 14 15\n"
                                " 0  1  2  3  4  5  6  7";

  test_symmetry_input(transform, expected_input_slice0, expected_input_slice1);
  test_symmetry_policy(transform, expected_policy);
}

void test_reflection_over_horizontal_with_rotation90() {
  othello::Tensorizor::ReflectionOverHorizontalWithRotation90Transform transform;

  std::string expected_input_slice0 = " 0  8 16 24 32 40 48 56\n"
                                      " 1  9 17 25 33 41 49 57\n"
                                      " 2 10 18 26 34 42 50 58\n"
                                      " 3 11 19 27 35 43 51 59\n"
                                      " 4 12 20 28 36 44 52 60\n"
                                      " 5 13 21 29 37 45 53 61\n"
                                      " 6 14 22 30 38 46 54 62\n"
                                      " 7 15 23 31 39 47 55 63";

  std::string expected_input_slice1 = " 64  72  80  88  96 104 112 120\n"
                                      " 65  73  81  89  97 105 113 121\n"
                                      " 66  74  82  90  98 106 114 122\n"
                                      " 67  75  83  91  99 107 115 123\n"
                                      " 68  76  84  92 100 108 116 124\n"
                                      " 69  77  85  93 101 109 117 125\n"
                                      " 70  78  86  94 102 110 118 126\n"
                                      " 71  79  87  95 103 111 119 127";

  std::string expected_policy = " 0  8 16 24 32 40 48 56\n"
                                " 1  9 17 25 33 41 49 57\n"
                                " 2 10 18 26 34 42 50 58\n"
                                " 3 11 19 27 28 43 51 59\n"
                                " 4 12 20 35 36 44 52 60\n"
                                " 5 13 21 29 37 45 53 61\n"
                                " 6 14 22 30 38 46 54 62\n"
                                " 7 15 23 31 39 47 55 63";

  test_symmetry_input(transform, expected_input_slice0, expected_input_slice1);
  test_symmetry_policy(transform, expected_policy);
}

void test_reflection_over_horizontal_with_rotation180() {
  othello::Tensorizor::ReflectionOverHorizontalWithRotation180Transform transform;

  std::string expected_input_slice0 = " 7  6  5  4  3  2  1  0\n"
                                      "15 14 13 12 11 10  9  8\n"
                                      "23 22 21 20 19 18 17 16\n"
                                      "31 30 29 28 27 26 25 24\n"
                                      "39 38 37 36 35 34 33 32\n"
                                      "47 46 45 44 43 42 41 40\n"
                                      "55 54 53 52 51 50 49 48\n"
                                      "63 62 61 60 59 58 57 56";

  std::string expected_input_slice1 = " 71  70  69  68  67  66  65  64\n"
                                      " 79  78  77  76  75  74  73  72\n"
                                      " 87  86  85  84  83  82  81  80\n"
                                      " 95  94  93  92  91  90  89  88\n"
                                      "103 102 101 100  99  98  97  96\n"
                                      "111 110 109 108 107 106 105 104\n"
                                      "119 118 117 116 115 114 113 112\n"
                                      "127 126 125 124 123 122 121 120";

  std::string expected_policy = " 7  6  5  4  3  2  1  0\n"
                                "15 14 13 12 11 10  9  8\n"
                                "23 22 21 20 19 18 17 16\n"
                                "31 30 29 27 28 26 25 24\n"
                                "39 38 37 35 36 34 33 32\n"
                                "47 46 45 44 43 42 41 40\n"
                                "55 54 53 52 51 50 49 48\n"
                                "63 62 61 60 59 58 57 56";

  test_symmetry_input(transform, expected_input_slice0, expected_input_slice1);
  test_symmetry_policy(transform, expected_policy);
}

void test_reflection_over_horizontal_with_rotation270() {
  othello::Tensorizor::ReflectionOverHorizontalWithRotation270Transform transform;

  std::string expected_input_slice0 = "63 55 47 39 31 23 15  7\n"
                                      "62 54 46 38 30 22 14  6\n"
                                      "61 53 45 37 29 21 13  5\n"
                                      "60 52 44 36 28 20 12  4\n"
                                      "59 51 43 35 27 19 11  3\n"
                                      "58 50 42 34 26 18 10  2\n"
                                      "57 49 41 33 25 17  9  1\n"
                                      "56 48 40 32 24 16  8  0";

  std::string expected_input_slice1 = "127 119 111 103  95  87  79  71\n"
                                      "126 118 110 102  94  86  78  70\n"
                                      "125 117 109 101  93  85  77  69\n"
                                      "124 116 108 100  92  84  76  68\n"
                                      "123 115 107  99  91  83  75  67\n"
                                      "122 114 106  98  90  82  74  66\n"
                                      "121 113 105  97  89  81  73  65\n"
                                      "120 112 104  96  88  80  72  64";

  std::string expected_policy = "63 55 47 39 31 23 15  7\n"
                                "62 54 46 38 30 22 14  6\n"
                                "61 53 45 37 29 21 13  5\n"
                                "60 52 44 27 28 20 12  4\n"
                                "59 51 43 35 36 19 11  3\n"
                                "58 50 42 34 26 18 10  2\n"
                                "57 49 41 33 25 17  9  1\n"
                                "56 48 40 32 24 16  8  0";

  test_symmetry_input(transform, expected_input_slice0, expected_input_slice1);
  test_symmetry_policy(transform, expected_policy);
}

void test_symmetries() {
  test_identity();
  test_rotation90();
  test_rotation180();
  test_rotation270();
  test_reflection_over_horizontal();
  test_reflection_over_horizontal_with_rotation90();
  test_reflection_over_horizontal_with_rotation180();
  test_reflection_over_horizontal_with_rotation270();
}

int main() {
  test_symmetries();

  if (global_fail_count > 0) {
    int total_count = global_pass_count + global_fail_count;
    printf("Failed %d of %d test%s!\n", global_fail_count, total_count, total_count > 1 ? "s" : "");
  } else {
    printf("All tests passed (%d of %d)!\n", global_pass_count, global_pass_count);
  }
  return 0;
}