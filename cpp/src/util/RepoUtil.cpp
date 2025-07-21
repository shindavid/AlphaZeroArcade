#include <util/RepoUtil.hpp>

#include <util/Exceptions.hpp>

namespace util {

Repo* Repo::instance_ = nullptr;

Repo* Repo::instance() {
  if (!instance_) {
    instance_ = new Repo();
  }
  return instance_;
}

boost::filesystem::path Repo::find_root() {
  boost::filesystem::path cwd = boost::filesystem::current_path();

  boost::filesystem::path path = cwd;
  while (true) {
    boost::filesystem::path marker = path / kMarkerFilename;
    if (boost::filesystem::is_regular_file(marker)) return path;
    if (path.has_parent_path()) {
      path = path.parent_path();
      continue;
    }
    break;
  }

  throw Exception("Could not find repo marker %s in any ancestor dir of %s", kMarkerFilename,
                  cwd.c_str());
}

}  // namespace util
