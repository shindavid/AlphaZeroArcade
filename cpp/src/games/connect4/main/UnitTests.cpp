#include <sstream>
#include <string>
#include <vector>

#include <core/tests/Common.hpp>
#include <games/connect4/Constants.hpp>
#include <games/connect4/Game.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

/*
 * Tests connect4 symmetry classes.
 */

using Game = c4::Game;
using State = Game::State;
using StateHistory = Game::StateHistory;
using PolicyTensor = Game::Types::PolicyTensor;
using IO = Game::IO;
using Rules = Game::Rules;

State make_init_state() {
  StateHistory history;
  history.initialize(Rules{});

  Rules::apply(history, 3);
  Rules::apply(history, 4);
  Rules::apply(history, 3);
  return history.current();
}

PolicyTensor make_policy(int move1, int move2) {
  PolicyTensor tensor;
  tensor.setZero();
  tensor(move1) = 1;
  tensor(move2) = 1;
  return tensor;
}

const std::string init_state_repr =
    "| | | | | | | |\n"
    "| | | | | | | |\n"
    "| | | | | | | |\n"
    "| | | | | | | |\n"
    "| | | |R| | | |\n"
    "| | | |R|Y| | |\n";

std::string get_repr(const State& state) {
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
  for (int i = 0; i < 6; ++i) {
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
  State state = make_init_state();

  group::element_t sym = groups::D1::kIdentity;
  group::element_t inv_sym = groups::D1::inverse(sym);
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

void test_flip() {
  State state = make_init_state();

  group::element_t sym = groups::D1::kFlip;
  group::element_t inv_sym = groups::D1::inverse(sym);
  Game::Symmetries::apply(state, sym);

  std::string repr = get_repr(state);
  std::string expected_repr =
      "| | | | | | | |\n"
      "| | | | | | | |\n"
      "| | | | | | | |\n"
      "| | | | | | | |\n"
      "| | | |R| | | |\n"
      "| | |Y|R| | | |\n";

  if (!validate_state(__func__, __LINE__, repr, expected_repr)) return;
  Game::Symmetries::apply(state, inv_sym);
  if (!validate_state(__func__, __LINE__, get_repr(state), init_state_repr)) return;

  PolicyTensor init_policy = make_policy(0, 1);
  PolicyTensor policy = init_policy;
  Game::Symmetries::apply(policy, sym);
  PolicyTensor expected_policy = make_policy(5, 6);
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
  test_flip();
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
