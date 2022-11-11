#pragma once

#include <boost/filesystem.hpp>

namespace util {

class Repo {
public:
  static constexpr const char* kMarkerFilename = "REPO_ROOT_MARKER";
  static boost::filesystem::path root() { return instance()->root_; }

private:
  Repo() : root_(find_root()) {}
  static Repo* instance();
  static boost::filesystem::path find_root();

  static Repo* instance_;
  const boost::filesystem::path root_;
};

}  // namespace util

#include <util/RepoUtilINLINES.cpp>
