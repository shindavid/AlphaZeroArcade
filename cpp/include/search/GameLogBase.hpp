#pragma once

#include "search/concepts/TraitsConcept.hpp"
#include "util/EigenUtil.hpp"

#include <cstdint>
#include <vector>

namespace search {

struct ShapeInfo {
  template <eigen_util::concepts::Shape Shape>
  void init(const char* nm, int target_idx);
  ~ShapeInfo();

  const char* name = nullptr;
  int* dims = nullptr;
  int num_dims = 0;
  int target_index = -1;
};

struct GameLogFileHeader {
  static constexpr uint16_t kCurrentVersion = 1;

  uint32_t num_games = 0;
  uint32_t num_rows = 0;
  uint16_t version = kCurrentVersion;
  uint8_t reserved[6] = {0};
};
static_assert(sizeof(GameLogFileHeader) == 16);

struct GameLogMetadata {
  uint64_t start_timestamp;
  uint32_t start_offset;  // relative to start of file
  uint32_t data_size;
  uint32_t num_samples;
  uint32_t num_positions;  // excludes terminal position
  uint32_t client_id;
  uint32_t reserved = 0;
};
static_assert(sizeof(GameLogMetadata) == 32);

class GameLogFileReader {
 public:
  GameLogFileReader(const char* buffer);

  const GameLogFileHeader& header() const;
  int num_games() const;

  int num_samples(int game_index) const;
  const char* game_data_buffer(int game_index) const;
  const GameLogMetadata& metadata(int game_index) const;
  const char* buffer() const { return buffer_; }

 private:
  const char* buffer_;
};

struct GameLogCommon {
  static constexpr int kAlignment = 16;

  using pos_index_t = int32_t;
  using mem_offset_t = int32_t;

  static void merge_files(const char** input_filenames, int n_input_filenames,
                          const char* output_filename);

  static constexpr int align(int offset);

  template <typename T>
  static int write_section(std::vector<char>& buf, const T* t, int count = 1, bool pad = true);
};

struct SparseTensorEntry {
  int32_t offset;
  float probability;
};

// tensor_encoding_t
//
// Used in TensorData. A value of t indicates that the TensorData::data field contains 4*abs(t)
// bytes of data.
//
// A negative value indicates that a dense tensor is stored, and a positive value indicates that a
// sparse tensor is stored.
using tensor_encoding_t = int32_t;

template <eigen_util::concepts::Shape Shape>
struct TensorData {
  using Tensor = eigen_util::FTensor<Shape>;
  static constexpr int kDenseCapacity = Tensor::Dimensions::total_size;
  static constexpr int kSparseCapacity = Tensor::Dimensions::total_size / 2;

  TensorData(bool valid, const Tensor&);
  int write_to(std::vector<char>& buf) const;
  int base_size() const { return sizeof(encoding) + 4 * std::abs(encoding); }
  int size() const;
  bool load(Tensor&) const;  // return true if valid tensor

  struct DenseData {
    float x[kDenseCapacity];
  };

  struct SparseData {
    SparseTensorEntry x[kSparseCapacity];
  };
  static_assert(sizeof(SparseData) == 8 * kSparseCapacity);

  union data_t {
    DenseData dense_repr;
    SparseData sparse_repr;
  };

  tensor_encoding_t encoding;
  data_t data;
};

template <search::concepts::Traits Traits>
struct GameLogBase : public GameLogCommon {
  using Game = Traits::Game;
  using State = Game::State;
  using PolicyShape = Game::Types::PolicyShape;
  using ActionValueShape = Game::Types::ActionValueShape;

  using GameLogFullRecord = Traits::GameLogFullRecord;
  using GameLogCompactRecord = Traits::GameLogCompactRecord;

  using full_record_vec_t = std::vector<GameLogFullRecord*>;

  using PolicyTensorData = TensorData<PolicyShape>;
  using ActionValueTensorData = TensorData<ActionValueShape>;

  static_assert(sizeof(PolicyTensorData) ==
                sizeof(tensor_encoding_t) + sizeof(typename PolicyTensorData::data_t));
  static_assert(sizeof(ActionValueTensorData) ==
                sizeof(tensor_encoding_t) + sizeof(typename ActionValueTensorData::data_t));
};

}  // namespace search

#include "inline/search/GameLogBase.inl"
