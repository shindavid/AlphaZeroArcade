#pragma once

#include <map>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

namespace util {

/*
 * API to access key-value pairs specified in <REPO_ROOT>/config.txt, where REPO_ROOT is the path
 * returned by util::Repo::root().
 *
 * TODO: currently, the python script main_loop.py copies the compiled c++ binary to a different
 * location, and launches it from there. The implicit promise is that the copied binary can be
 * launched from a cwd outside the repo, and also that its behavior is "frozen" - that it won't be
 * affected by changes to files inside the repo. Because of this config framework, that promise is
 * not true. We should clean this up, perhaps by copying the config file along with the compiled
 * binary. At present, the config file does not house any values that would affect the behavior of
 * the binary in any meaningful way, but this could change in the future.
 */
class Config {
 public:
  static constexpr const char* kFilename = "config.txt";

  static Config* instance();
  bool contains(const std::string& key) const;
  std::string get(const std::string& key) const;  // throws exception if key not found
  std::string get(const std::string& key, const std::string& default_value) const;
  boost::filesystem::path config_path() const { return config_path_; }

 private:
  Config();

  using map_t = std::map<std::string, std::string>;
  static Config* instance_;
  boost::filesystem::path config_path_;
  map_t map_;
};

}  // namespace util
