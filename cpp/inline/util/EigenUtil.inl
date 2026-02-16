#include "util/EigenUtil.hpp"

#include "util/Asserts.hpp"
#include "util/Random.hpp"
#include "util/StringUtil.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace eigen_util {

namespace detail {

// Helper function to round up at position n-1 if the (n)'th digit is 5 or greater
inline void round_up(std::string& s, size_t n) {
  while (n > 0 && (s[n - 1] == '9' || s[n - 1] == '.')) {
    if (s[n - 1] == '.') {
      --n;
      continue;
    }
    s[n - 1] = '0';
    --n;
  }
  if (n == 0) {
    // If rounding overflows all the way back to the start, prepend '1'
    s = '1' + s;
  } else {
    s[n - 1] += 1;  // Increment the digit at n-1
  }
}

/*
 * If s contains a decimal at-or-before the n'th character, chops off everything after the n'th
 * character, rounding up if appropriate. Then, removes all trailing zeros after the decimal.
 */
inline void trim(std::string& s, size_t n) {
  if (n >= s.size()) {
    // If n is beyond the length of the string, no trimming or rounding needed
    return;
  }

  // Check if we need to round up, based on the character at position n
  if (n < s.size() && s[n] >= '5') {
    round_up(s, n);
  }

  // Trim the string to ensure it has a maximum of n characters
  s = s.substr(0, n);

  // Remove trailing zeros and the decimal point, if needed
  size_t dot = s.find('.');
  if (dot != std::string::npos) {
    // Remove trailing zeros after the decimal point
    size_t last_non_zero = s.find_last_not_of('0');
    if (last_non_zero == dot) {
      // If the last non-zero character is the decimal point, remove it
      s = s.substr(0, dot);
    } else if (last_non_zero != std::string::npos) {
      // Trim after the last non-zero character
      s = s.substr(0, last_non_zero + 1);
    }
  }
}

/*
 * Counts the number of sig-digs in a numerical string.
 *
 * Assumes s is in standard notation (not scientific notation), without a leading + sign.
 *
 * Treats all leading AND trailing zeros as insignificant.
 */
inline int sigfigs(const std::string& s) {
  std::string t = s;
  boost::replace_all(t, "-", "");
  boost::replace_all(t, ".", "");
  int n = t.size();
  int a = 0;
  while (a < n && t.at(a) == '0') a++;

  if (a == n) return 0;

  int b = n - 1;
  while (b >= 0 && t.at(b) == '0') b--;

  return b - a + 1;
}

/*
 * Converts a float to a string of length at most 8.
 */
inline std::string float_to_str8(float x) {
  if (x == 0) return "";

  char buf[128];

  std::sprintf(buf, "%.8f", x);  // Standard
  std::string s(buf);
  trim(s, 8);

  std::sprintf(buf, "%.8e", x);  // Scientific
  std::string s2(buf);

  size_t e = s2.find('e');
  std::string mantissa = s2.substr(0, e);
  int exponent = std::atoi(s2.substr(e + 1).c_str());

  if (exponent == 0) {
    return s;
  }

  std::string exponent_str = std::to_string(exponent);

  int mantissa_capacity = 7 - exponent_str.size();
  trim(mantissa, mantissa_capacity);

  int scientific_sigfigs = sigfigs(mantissa);
  int standard_sigfigs = sigfigs(s);

  s2 = mantissa + "e" + exponent_str;
  if (s.size() > 8 || scientific_sigfigs > standard_sigfigs) {
    return s2;
  }
  return s;
}

template <typename T>
boost::json::array to_json(const T& array) {
  boost::json::array arr;
  for (int i = 0; i < array.size(); ++i) {
    arr.push_back(array.data()[i]);
  }
  return arr;
}

inline bool is_finite_safe(float x) {
    uint32_t u = std::bit_cast<uint32_t>(x);
    // Inifinity = 0x7F800000, NaN = any value > 0x7F800000
    return (u & 0x7FFFFFFF) < 0x7F800000;
}

template <typename T>
bool scan_buffer(const T* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (!is_finite_safe(data[i])) return false;
    }
    return true;
}

}  // namespace detail

template <typename Scalar>
template <typename Array, typename Urng, typename... DimTs>
Array UniformDirichletGen<Scalar>::generate(Urng&& urng, Scalar alpha, DimTs&&... dims) {
  static_assert(Array::MaxColsAtCompileTime > 0);
  static_assert(Array::MaxRowsAtCompileTime > 0);

  if (alpha != alpha_) {
    alpha_ = alpha;
    new (&gamma_) GammaGen(alpha);
  }

  Array out(dims...);
  for (int i = 0; i < out.size(); ++i) {
    out.data()[i] = gamma_.template generate<Eigen::Array<Scalar, 1, 1>>(1, 1, urng)(0);
  }
  out /= out.sum();
  return out;
}

template <typename Derived>
auto sort_columns(const Eigen::ArrayBase<Derived>& array, int row_ix, bool ascending) {
  RELEASE_ASSERT(row_ix < array.rows());

  using Column = std::remove_const_t<decltype(array.col(0).eval())>;
  int n = array.cols();
  Column columns[n];
  for (int i = 0; i < n; ++i) {
    columns[i] = array.col(i);
  }
  if (ascending) {
    std::sort(columns, columns + n,
              [row_ix](const Column& a, const Column& b) { return a(row_ix) < b(row_ix); });
  } else {
    std::sort(columns, columns + n,
              [row_ix](const Column& a, const Column& b) { return a(row_ix) > b(row_ix); });
  }

  auto out = array.eval();
  for (int i = 0; i < n; ++i) {
    out.col(i) = columns[i];
  }
  return out;
}

template <typename Derived>
auto sort_rows(const Eigen::ArrayBase<Derived>& array, int col_ix, bool ascending) {
  RELEASE_ASSERT(col_ix < array.cols());

  using Row = std::remove_const_t<decltype(array.row(0).eval())>;
  int n = array.rows();
  Row rows[n];
  for (int i = 0; i < n; ++i) {
    rows[i] = array.row(i);
  }
  if (ascending) {
    std::sort(rows, rows + n,
              [col_ix](const Row& a, const Row& b) { return a(col_ix) < b(col_ix); });
  } else {
    std::sort(rows, rows + n,
              [col_ix](const Row& a, const Row& b) { return a(col_ix) > b(col_ix); });
  }

  auto out = array.eval();
  for (int i = 0; i < n; ++i) {
    out.row(i) = rows[i];
  }
  return out;
}

template <class Derived>
void softmax_in_place(Eigen::ArrayBase<Derived>& a) {
  auto& x = static_cast<Derived&>(a);
  x = (x - x.maxCoeff()).exp();
  x = x / x.sum();
}

template <class Derived>
void softmax_in_place(Eigen::TensorBase<Derived, Eigen::WriteAccessors>& t) {
  auto& x = static_cast<Derived&>(t);
  x = (x - max(x)).exp();
  x = x / sum(x);
}

template <class Derived>
void rowwise_softmax_in_place(Eigen::TensorBase<Derived, Eigen::WriteAccessors>& t) {
  auto& x = static_cast<Derived&>(t);
  const Eigen::Index R = x.dimension(0);
  const Eigen::Index C = x.dimension(1);

  // Reduce across columns (axis = 1)
  const Eigen::array<int, 1> reduce_dim = {1};
  const Eigen::array<Eigen::Index, 2> col_vec = {R, 1};  // (R,1)
  const Eigen::array<Eigen::Index, 2> bcast = {1, C};    // -> (R,C)

  // 1) rowwise max (materialize!) -> reshape/broadcast
  const auto row_max = x.maximum(reduce_dim).eval();
  const auto row_max_bc = row_max.reshape(col_vec).broadcast(bcast);

  // 2) shift & exp
  const auto exp_shifted = (x - row_max_bc).exp();

  // 3) rowwise sum (materialize!) -> reshape/broadcast
  const auto row_sum = exp_shifted.sum(reduce_dim).eval();
  const auto row_sum_bc = row_sum.reshape(col_vec).broadcast(bcast);

  // 4) normalize
  x = exp_shifted / row_sum_bc;
}

template <class Derived>
void sigmoid_in_place(Eigen::TensorBase<Derived, Eigen::WriteAccessors>& t) {
  auto& x = static_cast<Derived&>(t);
  x = 1.0 / (1.0 + (-x).exp());
}

template <typename Derived>
inline int argmax(const Eigen::DenseBase<Derived>& arr) {
  static_assert(Derived::RowsAtCompileTime == 1 || Derived::ColsAtCompileTime == 1);
  Eigen::Index maxIndex;
  arr.maxCoeff(&maxIndex);
  return static_cast<int>(maxIndex);
}

template <typename Derived1, typename Derived2>
auto slice(const Eigen::DenseBase<Derived1>& data, const Eigen::DenseBase<Derived2>& indices) {
  static_assert((Derived2::RowsAtCompileTime == 1) || (Derived2::ColsAtCompileTime == 1));
  static_assert(std::is_integral_v<typename Derived2::Scalar>);

  using Scalar = Derived1::Scalar;
  constexpr int Cols = Derived1::ColsAtCompileTime;
  constexpr static int Options = Derived1::Options;
  using OutType = std::conditional_t<is_eigen_array<Derived1>::value,
                                     Eigen::Array<Scalar, Eigen::Dynamic, Cols, Options>,
                                     Eigen::Matrix<Scalar, Eigen::Dynamic, Cols, Options>>;
  Eigen::Index numIndices = indices.size();
  OutType out;
  out.resize(numIndices, Cols);
  for (Eigen::Index i = 0; i < numIndices; ++i) {
    out.row(i) = data.row(indices[i]);
  }
  return out;
}

template <concepts::FTensor Tensor>
auto reverse(const Tensor& tensor, int dim) {
  using Sizes = Tensor::Dimensions;
  constexpr int N = Sizes::count;
  static_assert(N > 0);

  Eigen::array<bool, N> rev;
  rev.fill(false);
  rev[dim] = true;
  return tensor.reverse(rev);
}

template <concepts::FTensor Tensor>
int sample(const Tensor& T) {
  const auto* data = T.data();
  int n = T.size();
  return util::Random::weighted_sample(data, data + n);
}

template <class Derived>
int size(const Eigen::TensorBase<Derived>& t) {
  constexpr int Rank = Derived::NumDimensions;

  const auto& x = static_cast<const Derived&>(t);
  auto dims = x.dimensions();
  int n = 1;
  for (int i = 0; i < Rank; ++i) n *= dims[i];
  return n;
}

template <concepts::FTensor Tensor>
bool normalize(Tensor& tensor, double eps) {
  auto s = sum(tensor);
  if (s < eps) return false;

  tensor = tensor / s;
  return true;
}

template <concepts::FTensor Tensor>
void randomly_zero_out(Tensor& tensor, int n) {
  using Shape = Tensor::Dimensions;
  constexpr size_t N = Shape::total_size;

  auto* data = tensor.data();
  util::Random::zero_out(data, data + N, n);
}

template <concepts::FTensor Tensor>
auto cwiseMax(const Tensor& tensor, float x) {
  Tensor out = tensor;
  auto& array = reinterpret_as_array(out);
  array = array.cwiseMax(x);
  return out;
}

template <concepts::FTensor Tensor>
const auto& reinterpret_as_array(const Tensor& tensor) {
  using Shape = Tensor::Dimensions;
  constexpr int N = Shape::total_size;
  using Array = FArray<N>;
  return reinterpret_cast<const Array&>(tensor);
}

template <concepts::FTensor Tensor>
auto& reinterpret_as_array(Tensor& tensor) {
  using Shape = Tensor::Dimensions;
  constexpr int N = Shape::total_size;
  using Array = FArray<N>;
  return reinterpret_cast<Array&>(tensor);
}

template <concepts::FArray Array>
const auto& reinterpret_as_tensor(const Array& array) {
  using Tensor = FTensor<Eigen::Sizes<extract_length_v<Array>>>;
  return reinterpret_cast<const Tensor&>(array);
}

template <concepts::FArray Array>
auto& reinterpret_as_tensor(Array& array) {
  using Tensor = FTensor<Eigen::Sizes<extract_length_v<Array>>>;
  return reinterpret_cast<Tensor&>(array);
}

template <typename T>
void debug_assert_is_valid_prob_distr(const T& distr, float eps) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;
  assert_is_valid_prob_distr(distr, eps);
}

template <concepts::FTensor Tensor>
void assert_is_valid_prob_distr(const Tensor& distr, float eps) {
  float s = sum(distr);
  float m = min(distr);

  if (m < 0 || abs(s - 1.0) > eps) {
    std::ostringstream ss;
    ss << distr;
    throw util::Exception("Invalid prob distr: sum={}, min={} distr:\n{}", s, m, ss.str());
  }
}

template <typename Derived>
void assert_is_valid_prob_distr(const Eigen::ArrayBase<Derived>& distr, float eps) {
  float s = distr.sum();
  float m = distr.minCoeff();

  if (m < 0 || abs(s - 1.0) > eps) {
    std::ostringstream ss;
    ss << distr;
    throw util::Exception("Invalid prob distr: sum={}, min={} distr:\n{}", s, m, ss.str());
  }
}

template <typename T>
void debug_validate_bounds(const T& t, float min, float max) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;
  validate_bounds(t, min, max);
}

template <typename T>
void validate_bounds(const T& t, float min, float max) {
  for (int i = 0; i < t.size(); ++i) {
    float v = t.data()[i];
    if (v < min || v > max) {
      std::ostringstream ss;
      ss << t;
      throw util::Exception("Value out of bounds [{}, {}]: {} distr:\n{}", min, max, v, ss.str());
    }
  }
}

template <typename Derived>
float sum(const Eigen::TensorBase<Derived>& tensor) {
  eigen_util::FTensor<Eigen::Sizes<>> out = tensor.sum();
  return out(0);
}

template <typename Derived>
float max(const Eigen::TensorBase<Derived>& tensor) {
  eigen_util::FTensor<Eigen::Sizes<>> out = tensor.maximum();
  return out(0);
}

template <typename Derived>
float min(const Eigen::TensorBase<Derived>& tensor) {
  eigen_util::FTensor<Eigen::Sizes<>> out = tensor.minimum();
  return out(0);
}

template <concepts::FTensor Tensor>
bool any(const Tensor& tensor) {
  const auto* data = tensor.data();
  for (int i = 0; i < tensor.size(); ++i) {
    if (data[i]) return true;
  }
  return false;
}

template <concepts::FTensor Tensor>
bool all(const Tensor& tensor) {
  const auto* data = tensor.data();
  for (int i = 0; i < tensor.size(); ++i) {
    if (!data[i]) return false;
  }
  return true;
}

template <concepts::FTensor Tensor>
int count(const Tensor& tensor) {
  int c = 0;
  for (int i = 0; i < tensor.size(); ++i) {
    c += bool(tensor.data()[i]);
  }
  return c;
}

template <typename Derived>
bool isfinite(const Eigen::DenseBase<Derived>& x) {
  return detail::scan_buffer(x.derived().data(), x.size());
}

template <concepts::FTensor Tensor>
bool isfinite(const Tensor& x) {
  return detail::scan_buffer(x.data(), x.dimensions().TotalSize());
}

template <concepts::FTensor Tensor>
bool equal(const Tensor& tensor1, const Tensor& tensor2) {
  Eigen::TensorFixedSize<bool, Eigen::Sizes<>, Eigen::RowMajor> out = (tensor1 == tensor2).all();
  return out();
}

template <concepts::FArray Array>
void left_rotate(Array& array, int n) {
  constexpr int N = extract_length_v<Array>;
  auto* data = array.data();
  std::rotate(data, data + n, data + N);
}

template <concepts::FArray Array>
void right_rotate(Array& array, int n) {
  constexpr int N = extract_length_v<Array>;
  auto* data = array.data();
  std::rotate(data, data + N - n, data + N);
}

template <concepts::FTensor Tensor>
void left_rotate(Tensor& tensor, int n) {
  using Shape = Tensor::Dimensions;
  constexpr int kRank = Shape::count;
  static_assert(kRank >= 1, "rotate_last_dim_left requires rank >= 1");

  constexpr int kTotalSize = Shape::total_size;
  auto* data = tensor.data();

  // ---- 1D fast path ----
  if constexpr (kRank == 1) {
    constexpr int N = kTotalSize;
    n = ((n % N) + N) % N;
    if (n == 0) return;
    std::rotate(data, data + n, data + N);
    return;
  }

  // ---- multi-D (RowMajor, last dimension varies fastest) ----
  constexpr int kInner = extract_dim_v<kRank - 1, Shape>;
  constexpr int kOuter = kTotalSize / kInner;  // number of row blocks

  n = ((n % kInner) + kInner) % kInner;
  if (n == 0) return;

  for (int i = 0; i < kOuter; ++i) {
    auto* row = data + i * kInner;
    std::rotate(row, row + n, row + kInner);
  }
}

template <concepts::FTensor Tensor>
void right_rotate(Tensor& tensor, int n) {
  using Shape = Tensor::Dimensions;
  constexpr int kRank = Shape::count;
  static_assert(kRank >= 1, "rotate_last_dim_left requires rank >= 1");

  constexpr int kTotalSize = Shape::total_size;
  auto* data = tensor.data();

  // ---- 1D fast path ----
  if constexpr (kRank == 1) {
    constexpr int N = kTotalSize;
    n = ((n % N) + N) % N;
    if (n == 0) return;
    std::rotate(data, data + N - n, data + N);
    return;
  }

  // ---- multi-D (RowMajor, last dimension varies fastest) ----
  constexpr int kInner = extract_dim_v<kRank - 1, Shape>;
  constexpr int kOuter = kTotalSize / kInner;  // number of row blocks

  n = ((n % kInner) + kInner) % kInner;
  if (n == 0) return;

  for (int i = 0; i < kOuter; ++i) {
    auto* row = data + i * kInner;
    std::rotate(row, row + kInner - n, row + kInner);
  }
}

namespace detail {

template <int Dim, class Scalar, class Stride>
using SliceMap =
  Eigen::Map<Eigen::Matrix<Scalar, Dim, Dim, Eigen::RowMajor>, Eigen::Unaligned, Stride>;

template <int Dim, concepts::FTensor Tensor, class F>
void transform_in_place(Tensor& tensor, F&& func) {
  using Scalar = Tensor::Scalar;
  using Shape = Tensor::Dimensions;

  constexpr int kDim0 = extract_dim_v<0, Shape>;
  constexpr int kTotalSize = Shape::total_size;
  constexpr int kTailStride = kTotalSize / kDim0;  // product of trailing dims

  static_assert(Dim * Dim <= kDim0, "Dim*Dim window exceeds first dimension length");

  auto* base_ptr = tensor.data();

  using Stride = Eigen::Stride<Dim * kTailStride, kTailStride>;

  for (int s = 0; s < kTailStride; ++s) {
    auto* slice_ptr = base_ptr + s;
    SliceMap<Dim, Scalar, Stride> block(slice_ptr, Dim, Dim);
    std::forward<F>(func)(block);
  }
}

}  // namespace detail

template <int Dim, concepts::FTensor Tensor>
void rot90_clockwise(Tensor& tensor) {
  detail::transform_in_place<Dim>(tensor, [](auto& slice) {
    slice.transposeInPlace();
    slice.rowwise().reverseInPlace();
  });
}

template <int Dim, concepts::FTensor Tensor>
void rot180(Tensor& tensor) {
  detail::transform_in_place<Dim>(tensor, [](auto& slice) {
    slice.rowwise().reverseInPlace();
    slice.colwise().reverseInPlace();
  });
}

template <int Dim, concepts::FTensor Tensor>
void rot270_clockwise(Tensor& tensor) {
  detail::transform_in_place<Dim>(tensor, [](auto& slice) {
    slice.transposeInPlace();
    slice.colwise().reverseInPlace();
  });
}

template <int Dim, concepts::FTensor Tensor>
void flip_vertical(Tensor& tensor) {
  detail::transform_in_place<Dim>(tensor, [](auto& slice) { slice.colwise().reverseInPlace(); });
}

template <int Dim, concepts::FTensor Tensor>
void mirror_horizontal(Tensor& tensor) {
  detail::transform_in_place<Dim>(tensor, [](auto& slice) { slice.rowwise().reverseInPlace(); });
}

template <int Dim, concepts::FTensor Tensor>
void flip_main_diag(Tensor& tensor) {
  detail::transform_in_place<Dim>(tensor, [](auto& slice) { slice.transposeInPlace(); });
}

template <int Dim, concepts::FTensor Tensor>
void flip_anti_diag(Tensor& tensor) {
  detail::transform_in_place<Dim>(tensor, [](auto& slice) {
    slice.transposeInPlace();
    slice.rowwise().reverseInPlace();
    slice.colwise().reverseInPlace();
  });
}

template <concepts::FTensor Tensor>
uint64_t hash(const Tensor& tensor) {
  using Scalar = Tensor::Scalar;
  constexpr int N = Tensor::Dimensions::total_size;
  return util::hash_memory<N * sizeof(Scalar)>(tensor.data());
}

template <typename Derived>
auto compute_covariance(const Eigen::MatrixBase<Derived>& X) {
  auto mean = X.colwise().mean();
  auto centered = X.rowwise() - mean;

  // Compute covariance matrix: (1/(n-1)) * (X-mu)^T * (X-mu)
  auto covariance = (centered.transpose() * centered) / (X.rows() - 1);
  return covariance;
}

template <typename Derived>
void print_array(std::ostream& os, const Eigen::ArrayBase<Derived>& array,
                 const std::vector<std::string>& column_names, const PrintArrayFormatMap* fmt_map) {
  int num_rows = array.rows();
  int num_cols = array.cols();

  RELEASE_ASSERT(num_cols == int(column_names.size()));

  std::vector<std::vector<std::tuple<std::string, int>>> str_width_pairs;
  std::vector<int> max_widths;
  str_width_pairs.reserve(num_cols);
  max_widths.reserve(num_cols);

  for (int j = 0; j < num_cols; ++j) {
    const std::string& column_name = column_names[j];
    int column_name_width = util::terminal_width(column_name);
    int max_width = column_name_width;

    bool using_default_func = true;
    std::function<std::string(float)> f = detail::float_to_str8;
    if (fmt_map) {
      auto it = fmt_map->find(column_name);
      if (it != fmt_map->end()) {
        f = it->second;
        using_default_func = false;
      }
    }

    std::vector<std::tuple<std::string, int>> col_pairs;
    col_pairs.reserve(num_rows + 1);
    col_pairs.push_back(std::make_tuple(column_name, column_name_width));
    for (int i = 0; i < num_rows; ++i) {
      float x = array(i, j);
      std::string s = f(x);
      int width;
      if (using_default_func) {
        width = s.size();  // avoid calling terminal_width for performance
      } else {
        width = util::terminal_width(s);
      }
      max_width = std::max(max_width, width);
      col_pairs.push_back(std::make_tuple(s, width));
    }

    str_width_pairs.push_back(col_pairs);
    max_widths.push_back(max_width);
  }

  for (int i = 0; i < num_rows + 1; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      const auto& tuple = str_width_pairs[j][i];
      std::string whitespace(max_widths[j] - std::get<1>(tuple) + (j > 0), ' ');
      os << whitespace << std::get<0>(tuple);
    }
    os << std::endl;
  }
}

template <typename Derived>
boost::json::object output_to_json(const Eigen::ArrayBase<Derived>& array,
                                   const std::vector<std::string>& key_strs,
                                   const PrintArrayFormatMap* fmt_map) {
  int num_rows = array.rows();
  int num_cols = array.cols();

  RELEASE_ASSERT(num_cols == int(key_strs.size()));

  boost::json::object obj;

  for (int j = 0; j < num_cols; ++j) {
    const std::string& key = key_strs[j];
    boost::json::array arr;

    if (fmt_map) {
      auto it = fmt_map->find(key);
      if (it != fmt_map->end()) {
        auto f = it->second;
        for (int i = 0; i < num_rows; ++i) {
          float x = array(i, j);
          std::string s = f(x);
          arr.emplace_back(boost::json::value(s));
        }
        obj[key] = arr;
        continue;
      }
    }

    for (int i = 0; i < num_rows; ++i) {
      float x = array(i, j);
      arr.emplace_back(x);
    }
    obj[key] = arr;
  }
  return obj;
}

template <typename Derived0, typename... Deriveds>
auto concatenate_columns(const Eigen::ArrayBase<Derived0>& first,
                         const Eigen::ArrayBase<Deriveds>&... rest) {
  static_assert((std::is_same_v<std::integral_constant<int, Derived0::ColsAtCompileTime>,
                                std::integral_constant<int, Deriveds::ColsAtCompileTime>> &&
                 ...));
  static_assert((std::is_same_v<std::integral_constant<int, Derived0::RowsAtCompileTime>,
                                std::integral_constant<int, Deriveds::RowsAtCompileTime>> &&
                 ...));
  static_assert((std::is_same_v<std::integral_constant<int, Derived0::Options>,
                                std::integral_constant<int, Deriveds::Options>> &&
                 ...));
  static_assert((std::is_same_v<std::integral_constant<int, Derived0::MaxRowsAtCompileTime>,
                                std::integral_constant<int, Deriveds::MaxRowsAtCompileTime>> &&
                 ...));
  static_assert((std::is_same_v<std::integral_constant<int, Derived0::MaxColsAtCompileTime>,
                                std::integral_constant<int, Deriveds::MaxColsAtCompileTime>> &&
                 ...));
  static_assert(Derived0::ColsAtCompileTime == 1);

  constexpr int num_arrays = sizeof...(rest) + 1;
  const int rows = first.rows();

  bool sizes_match = (... && (rest.rows() == rows));
  RELEASE_ASSERT(sizes_match, "All arrays must have the same number of rows");

  using Scalar = std::common_type_t<typename Derived0::Scalar, typename Deriveds::Scalar...>;
  constexpr int ColsAtCompileTime = num_arrays;
  constexpr int RowsAtCompileTime = Derived0::RowsAtCompileTime;
  constexpr int Options = Derived0::Options;
  constexpr int MaxRowsAtCompileTime = Derived0::MaxRowsAtCompileTime;

  using ResultT =
    Eigen::Array<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options, MaxRowsAtCompileTime>;

  ResultT result(rows, num_arrays);

  result.col(0) = first.template cast<Scalar>();
  int col_idx = 1;
  (..., (result.col(col_idx++) = rest.template cast<Scalar>()));  // Unpack and assign the arrays

  return result;
}

template <eigen_util::concepts::FTensor T>
boost::json::array to_json(const T& tensor) {
  return detail::to_json(tensor);
}

template <eigen_util::concepts::FArray T>
boost::json::array to_json(const T& array) {
  return detail::to_json(array);
}

template <concepts::FTensor Tensor>
Tensor zeros() {
  Tensor t;
  t.setZero();
  return t;
}

}  // namespace eigen_util
