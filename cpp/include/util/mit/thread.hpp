#pragma once

#include <thread>

#ifndef MIT_TEST_MODE
static_assert(false, "This file is not intended to be #include'd directly.");
#endif  // MIT_TEST_MODE

// TODO: implement MIT_TEST_MODE overrides

namespace mit {

using thread = std::thread;

}  // namespace mit
