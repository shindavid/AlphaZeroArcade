#pragma once

/*
 * In python we can bind sys.stdout to a StringIO object, thereby hijacking all future print()/sys.stdout.write()
 * calls so that those calls instead append to the StringIO.
 *
 * This module provides the mechanics to do something similar in c++.
 */
#include <cstdarg>
#include <iostream>
#include <sstream>

namespace util {

/*
 * Causes future xprintf() calls to dispatch to sprintf() and then push the resultant string to target.
 *
 * See xprintf() documentation.
 */
void set_xprintf_target(std::ostringstream& target);

/*
 * Causes future xprintf() calls to dispatch to printf().
 *
 * See xprintf() documentation.
 */
void clear_xprintf_target();

/*
 * If set_xprintf_target() was invoked at least once, and if clear_xprintf_target() was not called subsequently,
 * then this calls sprintf() to a char buffer and then pushes the resultant string to the target ostringstream.
 *
 * Else, simply dispatches to printf().
 */
int xprintf(char const* fmt, ...) __attribute__((format(printf, 1, 2)));

/*
 * Calls std::cout.flush() if no xprintf target is currently set.
 */
void xflush();

}  // namespace util

#include <util/inl/PrintUtil.inl>
