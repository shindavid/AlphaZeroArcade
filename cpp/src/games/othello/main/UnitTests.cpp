#include <sstream>
#include <string>
#include <vector>

#include <core/tests/Common.hpp>
#include <games/othello/Constants.hpp>
#include <games/othello/Game.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

/*
 * Tests othello symmetry classes.
 */

using Game = othello::Game;
using BaseState = Game::BaseState;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;

BaseState make_init_state() {
  StateHistory history;
  history.initialize(Rules{});

  Rules::apply(history, othello::kD3);
  return history.current();
}

PolicyTensor make_policy(int move) {
  PolicyTensor tensor;
  tensor.setZero();
  tensor(move) = 1;
  return tensor;
}

const std::string init_state_repr =
    "   A B C D E F G H\n"
    " 1| | | | | | | | |\n"
    " 2| | | | | | | | |\n"
    " 3| | |.|*|.| | | |\n"
    " 4| | | |*|*| | | |\n"
    " 5| | |.|*|0| | | |\n"
    " 6| | | | | | | | |\n"
    " 7| | | | | | | | |\n"
    " 8| | | | | | | | |\n";

std::string get_repr(const BaseState& state) {
  std::ostringstream ss;
  IO::print_state(ss, state);

  std::string s = ss.str();

  // only use the first 9 lines, we don't care about score part

  std::vector<std::string> lines;
  std::istringstream iss(s);
  std::string line;
  while (std::getline(iss, line)) {
    lines.push_back(line);
  }

  std::string repr;
  for (int i = 0; i < 9; ++i) {
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
  for (int c = 0; c < othello::kNumGlobalActions; ++c) {
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

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kA3);
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
      "   A B C D E F G H\n"
      " 1| | | | | | | | |\n"
      " 2| | | | | | | | |\n"
      " 3| | | |.| |.| | |\n"
      " 4| | | |*|*|*| | |\n"
      " 5| | | |0|*|.| | |\n"
      " 6| | | | | | | | |\n"
      " 7| | | | | | | | |\n"
      " 8| | | | | | | | |\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kF1);
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
      "   A B C D E F G H\n"
      " 1| | | | | | | | |\n"
      " 2| | | | | | | | |\n"
      " 3| | | | | | | | |\n"
      " 4| | | |0|*|.| | |\n"
      " 5| | | |*|*| | | |\n"
      " 6| | | |.|*|.| | |\n"
      " 7| | | | | | | | |\n"
      " 8| | | | | | | | |\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kH6);
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
      "   A B C D E F G H\n"
      " 1| | | | | | | | |\n"
      " 2| | | | | | | | |\n"
      " 3| | | | | | | | |\n"
      " 4| | |.|*|0| | | |\n"
      " 5| | |*|*|*| | | |\n"
      " 6| | |.| |.| | | |\n"
      " 7| | | | | | | | |\n"
      " 8| | | | | | | | |\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kC8);
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
      "   A B C D E F G H\n"
      " 1| | | | | | | | |\n"
      " 2| | | | | | | | |\n"
      " 3| | | | | | | | |\n"
      " 4| | |.|*|0| | | |\n"
      " 5| | | |*|*| | | |\n"
      " 6| | |.|*|.| | | |\n"
      " 7| | | | | | | | |\n"
      " 8| | | | | | | | |\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kA6);
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
      "   A B C D E F G H\n"
      " 1| | | | | | | | |\n"
      " 2| | | | | | | | |\n"
      " 3| | | |.|*|.| | |\n"
      " 4| | | |*|*| | | |\n"
      " 5| | | |0|*|.| | |\n"
      " 6| | | | | | | | |\n"
      " 7| | | | | | | | |\n"
      " 8| | | | | | | | |\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kH3);
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
      "   A B C D E F G H\n"
      " 1| | | | | | | | |\n"
      " 2| | | | | | | | |\n"
      " 3| | |.| |.| | | |\n"
      " 4| | |*|*|*| | | |\n"
      " 5| | |.|*|0| | | |\n"
      " 6| | | | | | | | |\n"
      " 7| | | | | | | | |\n"
      " 8| | | | | | | | |\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kC1);
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
      "   A B C D E F G H\n"
      " 1| | | | | | | | |\n"
      " 2| | | | | | | | |\n"
      " 3| | | | | | | | |\n"
      " 4| | | |0|*|.| | |\n"
      " 5| | | |*|*|*| | |\n"
      " 6| | | |.| |.| | |\n"
      " 7| | | | | | | | |\n"
      " 8| | | | | | | | |\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(othello::kA3);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(othello::kF8);
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
