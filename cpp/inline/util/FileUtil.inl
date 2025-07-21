#include "util/Exceptions.hpp"
#include "util/FileUtil.hpp"

#include <cstdio>

namespace util {

inline char* read_file(const char* filename, size_t file_size) {
  FILE* file = fopen(filename, "rb");
  if (!file) {
    throw util::Exception("Failed to open file '{}'", filename);
  }

  if (file_size == 0) {
    if (fseek(file, 0, SEEK_END) != 0) {
      fclose(file);
      throw util::Exception("Failed to seek to end of file '{}'", filename);
    }

    file_size = ftell(file);
    if (file_size < 0) {
      fclose(file);
      throw util::Exception("Failed to detect size of file '{}'", filename);
    }

    if (fseek(file, 0, SEEK_SET) != 0) {
      fclose(file);
      throw util::Exception("Failed to seek to start of file '{}'", filename);
    }
  }

  char* buffer = new char[file_size];
  size_t read_size = fread(buffer, 1, file_size, file);
  if (read_size != file_size) {
    delete[] buffer;
    fclose(file);
    throw util::Exception("Failed to read all bytes ({} != {}) of file '{}'", read_size, file_size,
                          filename);
  }

  fclose(file);
  return buffer;
}

}  // namespace util
