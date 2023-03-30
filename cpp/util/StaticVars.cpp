#include <util/AnsiCodes.hpp>
#include <util/Config.hpp>
#include <util/ParamDumper.hpp>
#include <util/Random.hpp>
#include <util/RepoUtil.hpp>
#include <util/ThreadSafePrinter.hpp>

namespace ansi {
Codes* Codes::instance_ = nullptr;
}  // namespace ansi

namespace util {

Config* Config::instance_ = nullptr;
ParamDumper* ParamDumper::instance_ = nullptr;
Random* Random::instance_ = nullptr;
Repo* Repo::instance_ = nullptr;
std::mutex ThreadSafePrinter::mutex_;

}  // namespace util
