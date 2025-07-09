#include <util/mit/functions.hpp>

#include <util/mit/scheduler.hpp>

namespace mit {

void seed(int s) {
  scheduler::instance().seed(s);
}

}  // namespace mit
