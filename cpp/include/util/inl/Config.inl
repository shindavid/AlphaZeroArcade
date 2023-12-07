#include <util/Config.hpp>

#include <fstream>

#include <boost/algorithm/string.hpp>

#include <util/Exception.hpp>
#include <util/RepoUtil.hpp>

namespace util {

inline Config* Config::instance() {
  if (!instance_) {
    instance_ = new Config();
  }
  return instance_;
}

inline bool Config::contains(const std::string& key) const { return map_.contains(key); }

inline std::string Config::get(const std::string& key) const {
  auto it = map_.find(key);
  if (it == map_.end()) {
    throw Exception("Mapping for key \"%s\" required in config file %s", key.c_str(),
                    config_path_.c_str());
  }
  return it->second;
}

inline std::string Config::get(const std::string& key, const std::string& default_value) const {
  auto it = map_.find(key);
  if (it == map_.end()) return default_value;
  return it->second;
}

inline Config::Config() : config_path_(Repo::root() / kFilename) {
  if (!boost::filesystem::is_regular_file(config_path_)) return;

  std::ifstream file(config_path_.c_str());
  std::string raw_line;
  while (std::getline(file, raw_line)) {
    std::string line = raw_line.substr(0, raw_line.rfind('#'));  // strip comment
    boost::algorithm::trim(line);
    if (line.empty()) continue;

    size_t eq = line.find('=');
    if (eq == std::string::npos) {
      throw Exception("Bad line in config file %s: %s", config_path_.c_str(), raw_line.c_str());
    }
    std::string key = line.substr(0, eq);
    std::string value = line.substr(eq + 1);

    boost::algorithm::trim(key);
    boost::algorithm::trim(value);

    if (contains(key)) {
      throw Exception("Duplicate key \"%s\" in config file %s", key.c_str(), config_path_.c_str());
    }
    map_[key] = value;
  }
}

}  // namespace util
