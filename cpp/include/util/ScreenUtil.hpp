#pragma once

namespace util {

int get_screen_width();

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
  static ScreenClearer& instance();

  bool ready_ = true;
};

}  // namespace util

#include "inline/util/ScreenUtil.inl"
