#pragma once

namespace util {

void clearscreen();

/*
 * Usage:
 *
 * ScreenClearer::clear_once();  // calls clearscreen()
 * ScreenClearer::clear_once();  // ignored!
 * ScreenClearer::reset();
 * ScreenClearer::clear_once();  // calls clearscreen()
 */
class ScreenClearer {
 public:
  static void clear_once();
  static void reset();

 private:
  static ScreenClearer* instance();

  static ScreenClearer* instance_;
  bool ready_ = true;
};

}  // namespace util

#include <util/inl/ScreenUtil.inl>
