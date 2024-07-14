#include <sstream>
#include <string>
#include <vector>

#include <core/tests/Common.hpp>
#include <games/tictactoe/Constants.hpp>
#include <games/tictactoe/Game.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

/*
 * Tests tictactoe symmetry classes.
 */

using Game = tictactoe::Game;
using BaseState = Game::BaseState;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;

BaseState make_init_state() {
  BaseState state;
  Rules::init_state(state);
  Rules::apply(state, 7);
  Rules::apply(state, 2);
  return state;
}

PolicyTensor make_policy(int move1, int move2) {
  PolicyTensor tensor;
  tensor.setZero();
  tensor(move1) = 1;
  tensor(move2) = 1;
  return tensor;
}

const std::string init_state_repr =
    "0 1 2  | | |O|\n"
    "3 4 5  | | | |\n"
    "6 7 8  | |X| |\n";

std::string get_repr(const BaseState& state) {
  std::ostringstream ss;
  IO::print_state(ss, state);

  std::string s = ss.str();
  std::vector<std::string> lines;
  std::istringstream iss(s);
  std::string line;
  while (std::getline(iss, line)) {
    lines.push_back(line);
  }

  std::string repr;
  for (int i = 0; i < 3; ++i) {
    repr += lines[i];
    repr += '\n';
  }

  return repr;
}

int global_pass_count = 0;
int global_fail_count = 0;

bool validate_state(const char* func, int line, const std::string& actual_repr,
                    const std::string& expected_repr) {
  if (actual_repr != expected_repr) {
    printf("Failure at %s():%d\n", func, line);
    std::cout << "Expected:" << std::endl;
    std::cout << expected_repr << std::endl;
    std::cout << "But got:" << std::endl;
    std::cout << actual_repr << std::endl;
    global_fail_count++;
    return false;
  } else {
    return true;
  }
}

bool validate_policy(const char* func, int line, const PolicyTensor& actual_policy,
                     const PolicyTensor& expected_policy) {
  bool failed = false;
  for (int c = 0; c < Game::Constants::kNumActions; ++c) {
    if (actual_policy(c) != expected_policy(c)) {
      if (!failed) {
        printf("Failure at %s():%d\n", func, line);
        failed = true;
      }
      printf("Expected policy(%s)=%f, but got %f\n", IO::action_to_str(c).c_str(),
             expected_policy(c), actual_policy(c));
    }
  }
  if (failed) {
    global_fail_count++;
    return false;
  }
  return true;
}

void test_identity() {
  BaseState state = make_init_state();

  group::element_t sym = groups::D4::kIdentity;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr = init_state_repr;

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(0, 1);
  if (!validate_policy(__func__, __LINE__, policy, expected_policy)) return;
  Game::Symmetries::apply(policy, inv_sym);
  if (!validate_policy(__func__, __LINE__, policy, init_policy)) return;

  printf("Success: %s()\n", __func__);
  global_pass_count++;
}

void test_rot90_clockwise() {
  BaseState state = make_init_state();

  group::element_t sym = groups::D4::kRot90;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
      "0 1 2  | | | |\n"
      "3 4 5  |X| | |\n"
      "6 7 8  | | |O|\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(2, 5);
  if (!validate_policy(__func__, __LINE__, policy, expected_policy)) return;
  Game::Symmetries::apply(policy, inv_sym);
  if (!validate_policy(__func__, __LINE__, policy, init_policy)) return;

  printf("Success: %s()\n", __func__);
  global_pass_count++;
}

void test_rot180() {
  BaseState state = make_init_state();

  group::element_t sym = groups::D4::kRot180;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
      "0 1 2  | |X| |\n"
      "3 4 5  | | | |\n"
      "6 7 8  |O| | |\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(7, 8);
  if (!validate_policy(__func__, __LINE__, policy, expected_policy)) return;
  Game::Symmetries::apply(policy, inv_sym);
  if (!validate_policy(__func__, __LINE__, policy, init_policy)) return;

  printf("Success: %s()\n", __func__);
  global_pass_count++;
}

void test_rot270_clockwise() {
  BaseState state = make_init_state();

  group::element_t sym = groups::D4::kRot270;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
      "0 1 2  |O| | |\n"
      "3 4 5  | | |X|\n"
      "6 7 8  | | | |\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(3, 6);
  if (!validate_policy(__func__, __LINE__, policy, expected_policy)) return;
  Game::Symmetries::apply(policy, inv_sym);
  if (!validate_policy(__func__, __LINE__, policy, init_policy)) return;

  printf("Success: %s()\n", __func__);
  global_pass_count++;
}

void test_flip_vertical() {
  BaseState state = make_init_state();

  group::element_t sym = groups::D4::kFlipVertical;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
      "0 1 2  | |X| |\n"
      "3 4 5  | | | |\n"
      "6 7 8  | | |O|\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(6, 7);
  if (!validate_policy(__func__, __LINE__, policy, expected_policy)) return;
  Game::Symmetries::apply(policy, inv_sym);
  if (!validate_policy(__func__, __LINE__, policy, init_policy)) return;

  printf("Success: %s()\n", __func__);
  global_pass_count++;
}

void test_mirror_horizontal() {
  BaseState state = make_init_state();

  group::element_t sym = groups::D4::kMirrorHorizontal;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
      "0 1 2  |O| | |\n"
      "3 4 5  | | | |\n"
      "6 7 8  | |X| |\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(1, 2);
  if (!validate_policy(__func__, __LINE__, policy, expected_policy)) return;
  Game::Symmetries::apply(policy, inv_sym);
  if (!validate_policy(__func__, __LINE__, policy, init_policy)) return;

  printf("Success: %s()\n", __func__);
  global_pass_count++;
}

void test_flip_main_diag() {
  BaseState state = make_init_state();

  group::element_t sym = groups::D4::kFlipMainDiag;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
      "0 1 2  | | | |\n"
      "3 4 5  | | |X|\n"
      "6 7 8  |O| | |\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(0, 3);
  if (!validate_policy(__func__, __LINE__, policy, expected_policy)) return;
  Game::Symmetries::apply(policy, inv_sym);
  if (!validate_policy(__func__, __LINE__, policy, init_policy)) return;

  printf("Success: %s()\n", __func__);
  global_pass_count++;
}

void test_flip_anti_diag() {
  BaseState state = make_init_state();

  group::element_t sym = groups::D4::kFlipAntiDiag;
  group::element_t inv_sym = groups::D4::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
      "0 1 2  | | |O|\n"
      "3 4 5  |X| | |\n"
      "6 7 8  | | | |\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(5, 8);
  if (!validate_policy(__func__, __LINE__, policy, expected_policy)) return;
  Game::Symmetries::apply(policy, inv_sym);
  if (!validate_policy(__func__, __LINE__, policy, init_policy)) return;

  printf("Success: %s()\n", __func__);
  global_pass_count++;
}

void test_action_transforms() {
  if (core::tests::Common<Game>::test_action_transforms(__func__) == false) {
    global_fail_count++;
    return;
  } else {
    printf("Success: %s()\n", __func__);
    global_pass_count++;
  }
}

void test_canonicalization() {
  BaseState state;
  Rules::init_state(state);
  Rules::apply(state, 2);
  Rules::apply(state, 1);

  std::string expected_repr =
      "0 1 2  | |O|X|\n"
      "3 4 5  | | | |\n"
      "6 7 8  | | | |\n";

  std::string repr = get_repr(state);

  if (repr != expected_repr) {
    printf("Failure in %s() at %s:%d\n", __func__, __FILE__, __LINE__);
    printf("Expected:\n%s\n", expected_repr.c_str());
    printf("But got:\n%s\n", repr.c_str());
    global_fail_count++;
    return;
  }

  group::element_t e = Game::Symmetries::get_canonical_symmetry(state);
  if (e != groups::D4::kMirrorHorizontal) {
    printf("Failure in %s() at %s:%d\n", __func__, __FILE__, __LINE__);
    printf("Expected canonical symmetry: %d, but got %d\n", groups::D4::kMirrorHorizontal, e);
    global_fail_count++;
    return;
  }

  Game::Symmetries::apply(state, e);

  expected_repr =
      "0 1 2  |X|O| |\n"
      "3 4 5  | | | |\n"
      "6 7 8  | | | |\n";

  repr = get_repr(state);

  if (repr != expected_repr) {
    printf("Failure in %s() at %s:%d\n", __func__, __FILE__, __LINE__);
    printf("Expected:\n%s\n", expected_repr.c_str());
    printf("But got:\n%s\n", repr.c_str());
    global_fail_count++;
    return;
  }

  printf("Success: %s()\n", __func__);
  global_pass_count++;
}

void test_symmetries() {
  test_identity();
  test_rot90_clockwise();
  test_rot180();
  test_rot270_clockwise();
  test_flip_vertical();
  test_mirror_horizontal();
  test_flip_main_diag();
  test_flip_anti_diag();
  test_action_transforms();
  test_canonicalization();
}

int main() {
  util::set_tty_mode(false);
  test_symmetries();

  if (global_fail_count > 0) {
    int total_count = global_pass_count + global_fail_count;
    printf("Failed %d of %d test%s!\n", global_fail_count, total_count, total_count > 1 ? "s" : "");
  } else {
    printf("All tests passed (%d of %d)!\n", global_pass_count, global_pass_count);
  }
  return global_fail_count ? 1 : 0;
}
