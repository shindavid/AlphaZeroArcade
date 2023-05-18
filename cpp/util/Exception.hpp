#pragma once

#include <cstdarg>
#include <iostream>

namespace util {

/*
 * Like std::runtime_error, but allows printf-style arguments.
 *
 * TODO: dynamic resizing in case the error text exceeds 1024 chars.
 *
 * TODO: once the std::format library is implemented in gcc, use that instead of printf-style formatting.
 */
class Exception : public std::exception {
public:
  Exception(char const* fmt, ...) __attribute__((format(printf, 2, 3)));
  char const* what() const throw() { return text_; }

private:
  char text_[1024];
};

/*
 * A variant of util::Exception.
 *
 * Use util::CleanException when you want to throw an exception that is not due to a bug in the program. For example, if
 * the user passes invalid cmdline args, then you should throw a util::CleanException. The main() of the program
 * should catch this exception and print the error message to stderr. This avoids the generation of unnecessary
 * core dumps. The descriptor "clean" comes from the fact that unwanted core-dump files are "dirty", so an exception
 * that doesn't produce them is "clean".
 *
 * TODO: come up with a better descriptor than "clean".
 *
 * NOTE: I would inherit from Exception if I could, but there are some technical issues with that (-Wformat-security).
 */
class CleanException : public std::exception {
public:
  CleanException(char const* fmt, ...) __attribute__((format(printf, 2, 3)));
  char const* what() const throw() { return text_; }

private:
  char text_[1024];
};

/*
 * Like assert(), with 2 notable differences:
 *
 * 1. Evaluated even if NDEBUG is defined.
 * 2. Throws a CleanException instead of aborting the program. See CleanException documentation.
 */
void clean_assert(bool condition, char const* fmt, ...) __attribute__((format(printf, 2, 3)));

}  // namespace util

#include <util/inl/Exception.inl>
