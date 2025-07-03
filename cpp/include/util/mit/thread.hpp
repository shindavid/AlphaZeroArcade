#pragma once

#ifdef MIT_TEST_MODE

// TODO: implement MIT_TEST_MODE overrides

#include <thread>

namespace mit {
using thread = std::thread;
}  // namespace mit

#else  // MIT_TEST_MODE

#include <thread>

namespace mit {
using thread = std::thread;
}  // namespace mit

#endif  // MIT_TEST_MODE
