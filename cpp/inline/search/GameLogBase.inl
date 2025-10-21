#include "search/GameLogBase.hpp"

#include "util/Asserts.hpp"
#include "util/EigenUtil.hpp"
#include "util/FileUtil.hpp"
#include "util/Math.hpp"

#include <algorithm>

namespace search {

template <eigen_util::concepts::Shape Shape>
void ShapeInfo::init(const char* nm, int target_idx) {
  this->name = nm;
  this->dims = new int[Shape::count];
  this->num_dims = Shape::count;
  this->target_index = target_idx;

  Shape shape;
  for (int i = 0; i < Shape::count; ++i) {
    this->dims[i] = shape[i];
  }
}

inline ShapeInfo::~ShapeInfo() { delete[] dims; }

inline GameLogFileReader::GameLogFileReader(const char* buffer) : buffer_(buffer) {}

inline const GameLogFileHeader& GameLogFileReader::header() const {
  return *reinterpret_cast<const GameLogFileHeader*>(buffer_);
}

inline int GameLogFileReader::num_games() const { return header().num_games; }

inline int GameLogFileReader::num_samples(int game_index) const {
  return metadata(game_index).num_samples;
}

inline const char* GameLogFileReader::game_data_buffer(int game_index) const {
  const GameLogMetadata& m = metadata(game_index);
  return buffer_ + m.start_offset;
}

inline const GameLogMetadata& GameLogFileReader::metadata(int game_index) const {
  const GameLogFileHeader& h = header();
  constexpr auto m = sizeof(GameLogMetadata);
  return *reinterpret_cast<const GameLogMetadata*>(buffer_ + sizeof(h) + game_index * m);
}

inline void GameLogCommon::merge_files(const char** input_filenames, int n_input_filenames,
                                       const char* output_filename) {
  std::vector<GameLogFileReader> readers;

  for (int i = 0; i < n_input_filenames; ++i) {
    char* buffer = util::read_file(input_filenames[i]);
    readers.emplace_back(buffer);
  }

  GameLogFileHeader header;
  for (const auto& reader : readers) {
    header.num_games += reader.num_games();
    header.num_rows += reader.header().num_rows;
  }

  struct game_t {
    int reader_index;
    int game_index;
  };

  uint64_t total_data_size = 0;
  std::vector<game_t> games;
  games.reserve(header.num_games);
  for (int i = 0; i < n_input_filenames; ++i) {
    int num_games = readers[i].num_games();
    for (int j = 0; j < num_games; ++j) {
      games.push_back(game_t{i, j});
      total_data_size += readers[i].metadata(j).data_size;
    }
  }

  std::sort(games.begin(), games.end(), [&](const game_t& a, const game_t& b) {
    return readers[a.reader_index].metadata(a.game_index).start_timestamp <
           readers[b.reader_index].metadata(b.game_index).start_timestamp;
  });

  uint32_t start_offset = sizeof(GameLogFileHeader) + sizeof(GameLogMetadata) * header.num_games;
  size_t out_buffer_size = start_offset + total_data_size;

  std::vector<char> out_buffer;
  out_buffer.reserve(out_buffer_size);

  const char* h = reinterpret_cast<const char*>(&header);
  out_buffer.insert(out_buffer.end(), h, h + sizeof(header));

  for (const game_t& game : games) {
    const GameLogFileReader& reader = readers[game.reader_index];
    GameLogMetadata metadata = reader.metadata(game.game_index);

    metadata.start_offset = start_offset;
    start_offset += metadata.data_size;

    const char* m = reinterpret_cast<const char*>(&metadata);
    out_buffer.insert(out_buffer.end(), m, m + sizeof(metadata));
  }

  for (const game_t& game : games) {
    const GameLogFileReader& reader = readers[game.reader_index];
    GameLogMetadata metadata = reader.metadata(game.game_index);
    const char* game_buffer = reader.game_data_buffer(game.game_index);
    out_buffer.insert(out_buffer.end(), game_buffer, game_buffer + metadata.data_size);
  }

  RELEASE_ASSERT(out_buffer.size() == out_buffer_size, "Size mismatch {} != {}", out_buffer.size(),
                 out_buffer_size);

  std::ofstream output_file(output_filename, std::ios::binary);
  output_file.write(out_buffer.data(), out_buffer.size());
  RELEASE_ASSERT(output_file.good(), "Failed to write to {}", output_filename);
  output_file.close();

  for (const auto& reader : readers) {
    delete[] reader.buffer();
  }
}

inline constexpr int GameLogCommon::align(int offset) {
  return math::round_up_to_nearest_multiple(offset, kAlignment);
}

template <typename T>
int GameLogCommon::write_section(std::vector<char>& buf, const T* t, int count, bool pad) {
  constexpr int A = kAlignment;
  int n_bytes = sizeof(T) * count;
  const char* bytes = reinterpret_cast<const char*>(t);
  buf.insert(buf.end(), bytes, bytes + n_bytes);

  if (!pad) return n_bytes;

  int remainder = n_bytes % A;
  int padding = 0;
  if (remainder) {
    padding = A - remainder;
    uint8_t zeroes[A] = {0};
    buf.insert(buf.end(), zeroes, zeroes + padding);
  }
  return n_bytes + padding;
}

template <eigen_util::concepts::Shape Shape>
TensorData<Shape>::TensorData(bool valid, const Tensor& tensor) {
  if (!valid) {
    encoding = 0;
    return;
  }

  constexpr int N = Tensor::Dimensions::total_size;
  int num_nonzero_entries = eigen_util::count(tensor);

  const auto* src = tensor.data();
  if (num_nonzero_entries <= kSparseCapacity) {
    int n = num_nonzero_entries;
    encoding = 2 * n;
    auto* dst = data.sparse_repr.x;

    int w = 0;
    for (int r = 0; r < N; ++r) {
      if (src[r]) {
        dst[w].offset = r;
        dst[w].probability = src[r];
        ++w;
      }
    }
  } else {
    encoding = -N;
    auto* dst = data.dense_repr.x;
    std::copy(src, src + N, dst);
  }
}

template <eigen_util::concepts::Shape Shape>
int TensorData<Shape>::write_to(std::vector<char>& buf) const {
  int b = base_size();
  int s = size();
  const char* bytes = reinterpret_cast<const char*>(this);
  buf.insert(buf.end(), bytes, bytes + b);
  for (int i = b; i < s; ++i) {
    buf.push_back(0);
  }
  return s;
}

template <eigen_util::concepts::Shape Shape>
int TensorData<Shape>::size() const {
  return math::round_up_to_nearest_multiple(base_size(), GameLogCommon::kAlignment);
}

template <eigen_util::concepts::Shape Shape>
bool TensorData<Shape>::load(Tensor& tensor) const {
  if (encoding == 0) {
    // no policy target
    return false;
  }

  if (encoding < 0) {
    // dense format
    int n = -encoding;
    const float* src = &data.dense_repr.x[0];
    float* dst = tensor.data();
    std::copy(src, src + n, dst);
  } else {
    // sparse format
    int n = encoding / 2;
    tensor.setZero();

    for (int i = 0; i < n; ++i) {
      SparseTensorEntry s = data.sparse_repr.x[i];
      tensor.data()[s.offset] = s.probability;
    }
  }
  return true;
}

}  // namespace search
