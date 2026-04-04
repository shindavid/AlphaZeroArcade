#pragma once

namespace core {

template <typename InputEncoder_, typename PolicyEncoding_>
struct TensorEncodings {
  using InputEncoder = InputEncoder_;
  using PolicyEncoding = PolicyEncoding_;
};

}  // namespace core
