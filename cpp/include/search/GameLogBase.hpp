#pragma once

#include "search/concepts/TraitsConcept.hpp"
#include "util/EigenUtil.hpp"

#include <cstdint>
#include <vector>

namespace search {

struct ShapeInfo {
  template <eigen_util::concepts::FTensor Tensor>
  void init(const char* nm, int target_idx, bool primary);
  ~ShapeInfo();

  const char* name = nullptr;
  int* dims = nullptr;
  int num_dims = 0;
  int target_index = -1;
  int is_primary = false;
};

struct GameLogFileHeader {
  uint32_t num_games = 0;
  uint32_t num_rows = 0;
  uint64_t reserved = 0;
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

  // tensor_encoding_t
  //
  // Used in TensorData. A value of t indicates that the TensorData::data field contains 4*abs(t)
  // bytes of data.
  //
  // A negative value indicates that a dense tensor is stored, and a positive value indicates that a
  // sparse tensor is stored.
  using tensor_encoding_t = int32_t;

  struct SparseTensorEntry {
    int32_t offset;
    float probability;
  };

  static void merge_files(const char** input_filenames, int n_input_filenames,
                          const char* output_filename);

  static constexpr int align(int offset);

  template <typename T>
  static int write_section(std::vector<char>& buf, const T* t, int count = 1, bool pad = true);
};

template <search::concepts::Traits Traits>
struct GameLogBase : public GameLogCommon {
  using Game = Traits::Game;
  using State = Game::State;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;

  using GameLogFullRecord = Traits::GameLogFullRecord;
  using GameLogCompactRecord = Traits::GameLogCompactRecord;

  using full_record_vec_t = std::vector<GameLogFullRecord*>;

  struct TensorData {
    static constexpr int kDenseCapacity = PolicyTensor::Dimensions::total_size;
    static constexpr int kSparseCapacity = PolicyTensor::Dimensions::total_size / 2;

    TensorData(bool valid, const PolicyTensor&);
    int write_to(std::vector<char>& buf) const;
    int size() const { return sizeof(encoding) + 4 * std::abs(encoding); }
    bool load(PolicyTensor&) const;  // return true if valid tensor

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

  static_assert(sizeof(TensorData) ==
                sizeof(tensor_encoding_t) + sizeof(typename TensorData::data_t));
};

}  // namespace search

#include "inline/search/GameLogBase.inl"
