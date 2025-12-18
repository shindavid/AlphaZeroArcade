#pragma once

namespace util {

// A simple 1D Gaussian representation.
//
// Capable of representing degenerate cases such as +inf or -inf mean.
class Gaussian1D {
 public:
  // Special values for representing +inf or -inf mean
  // Since a variance must be non-negative, we use the variance_ field to indicate special values.
  // This overloading nicely limits the size of the class to 8 bytes.
  static constexpr float kVarianceNegInf = -1;
  static constexpr float kVariancePosInf = -3;
  static constexpr float kVarianceUnset = -2;

  Gaussian1D(float m, float v) : mean_(m), variance_(v) {}
  Gaussian1D() = default;

  static Gaussian1D neg_inf() { return Gaussian1D(0, kVarianceNegInf); }
  static Gaussian1D pos_inf() { return Gaussian1D(0, kVariancePosInf); }

  bool operator==(const Gaussian1D& other) const = default;

  // unary-negative operator:
  Gaussian1D operator-() const {
    return Gaussian1D(-mean_, variance_ >= 0 ? variance_ : -4 - variance_);
  }

  bool valid() const { return variance_ != kVarianceUnset; }
  float mean() const { return mean_; }          // assumes valid()
  float variance() const { return variance_; }  // assumes valid()

 private:
  float mean_ = 0;                   // Must be 0 if variance_ is special value.
  float variance_ = kVarianceUnset;  // Negative indicates special value.
};

}  // namespace util
