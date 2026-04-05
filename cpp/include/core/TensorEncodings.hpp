#pragma once

namespace core {

template <typename InputEncoder_, typename PolicyEncoding_, typename GameResultEncoding_>
struct TensorEncodings {
  using InputEncoder = InputEncoder_;
  using PolicyEncoding = PolicyEncoding_;
  using GameResultEncoding = GameResultEncoding_;
};

}  // namespace core
