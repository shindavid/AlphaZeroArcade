#pragma once

#include <cstdarg>
#include <exception>
#include <format>

namespace util {

/*
 * Like std::runtime_error, but with std::format() mechanics.
 */
class Exception : public std::exception {
 public:
  Exception() : std::exception() {}

  template <typename... Ts>
  Exception(std::format_string<Ts...> fmt, Ts&&... ts) : std::exception() {
    what_ = std::format(fmt, std::forward<Ts>(ts)...);
  }
  char const* what() const throw() { return what_.c_str(); }

 private:
  std::string what_;
};

/*
 * A variant of util::Exception.
 *
 * Use util::CleanException when you want to throw an exception that is not due to a bug in the
 * program. For example, if the user passes invalid cmdline args, then you should throw a
 * util::CleanException. The main() of the program should catch this exception and print the error
 * message to stderr. This avoids the generation of unnecessary core dumps. The descriptor "clean"
 * comes from the fact that unwanted core-dump files are "dirty", so an exception that doesn't
 * produce them is "clean".
 */
class CleanException : public Exception {
 public:
  using Exception::Exception;
};

}  // namespace util
