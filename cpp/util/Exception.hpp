#pragma once

/*
 * Like std::runtime_error, but allows printf-style formatting.
 */
#include <cstdarg>
#include <iostream>

namespace util {

/*
 * TODO: dynamic resizing in case the error text exceeds 1024 chars.
 */
class Exception : public std::exception {
public:
  Exception(char const* fmt, ...) __attribute__((format(printf, 2, 3))) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(text_, sizeof(text_), fmt, ap);
    va_end(ap);
  }

  char const* what() const throw() { return text_; }

private:
  char text_[1024];
};

}  // namespace util
