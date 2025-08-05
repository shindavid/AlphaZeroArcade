#include "util/OsUtil.hpp"

#include "util/Asserts.hpp"
#include "util/Exceptions.hpp"
#include "util/LoggingUtil.hpp"
#include "util/StringUtil.hpp"

#include <boost/process.hpp>
#include <boost/filesystem.hpp>

#include <format>
#include <set>
#include <string>
#include <vector>

namespace os_util {

inline void free_port(int port) {
  namespace bp = boost::process;

  // Find the process using the port
  std::string cmd = std::format("lsof -i :{}", port);
  bp::ipstream pipe_stream;
  bp::child c(cmd, bp::std_out > pipe_stream);

  std::string line;
  int pid_column = -1;
  std::set<int> pids;
  while (pipe_stream && std::getline(pipe_stream, line) && !line.empty()) {
    if (pid_column == -1) {
      std::vector<std::string> columns = util::split(line);
      for (size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] == "PID") {
          pid_column = i;
          break;
        }
      }
      RELEASE_ASSERT(pid_column != -1, "Failed to find PID column in header line: {}", line);
      continue;
    }

    std::vector<std::string> columns = util::split(line);
    int pid = std::stoi(columns[pid_column]);
    pids.insert(pid);
  }
  c.wait();

  if (pids.empty()) return;

  for (int pid : pids) {
    LOG_WARN("Port {} is unavailable! Currently used by process {}. Killing...", port, pid);
    bp::system(std::format("kill -9 {}", pid));
  }

  // Wait a moment for the OS to release the port
  int n_retries = 5;
  auto retry_delay = std::chrono::milliseconds(100);
  for (int i = 0; i < n_retries; ++i) {
    std::this_thread::sleep_for(retry_delay);
    if (bp::system(std::format("lsof -i :{}", port)) != 0) {
      LOG_INFO("Port {} is now free", port);
      return;
    }
  }

  throw util::Exception("Failed to free port {}", port);
}

}  // namespace os_util
