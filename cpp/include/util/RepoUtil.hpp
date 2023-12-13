#pragma once

#include <boost/filesystem.hpp>

namespace util {

/*
 * util::Repo::root() returns the path to the root of the repo.
 *
 * This assumes that your cwd is somewhere inside the repo. It works by navigating up the directory
 * tree until it finds the REPO_ROOT_MARKER file, which lives at the root of the repo.
 *
 * See TODO comment at the top of cpp/util/Config.hpp
 */
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
