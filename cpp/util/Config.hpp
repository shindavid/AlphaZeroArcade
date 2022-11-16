#pragma once

#include <map>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

namespace util {

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

#include <util/inl/Config.inl>
