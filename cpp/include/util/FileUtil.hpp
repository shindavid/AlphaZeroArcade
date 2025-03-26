#pragma once

#include <cstddef>

namespace util {

// Read the contents of a file into a buffer. The buffer is allocated with new[] and must be deleted
// by the caller.
//
// If file_size is 0, the file size is determined by seeking to the end of the file and then
// seeking back to the start. If file_size is non-zero, it is used as the file size.
//
// If there is an error reading the file, throws a util::Exception.
char* read_file(const char* filename, size_t file_size = 0);

}  // namespace util

#include <inline/util/FileUtil.inl>
