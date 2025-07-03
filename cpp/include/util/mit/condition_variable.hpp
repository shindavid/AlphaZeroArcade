#pragma once

#ifdef MIT_TEST_MODE

// TODO: implement MIT_TEST_MODE overrides

#include <condition_variable>

namespace mit {
using condition_variable = std::condition_variable;
}  // namespace mit

#else  // MIT_TEST_MODE

#include <condition_variable>

namespace mit {
using condition_variable = std::condition_variable;
}  // namespace mit

#endif  // MIT_TEST_MODE
