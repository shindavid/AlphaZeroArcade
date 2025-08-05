#pragma once

namespace os_util {

// If the port is in use, forcibly free it by killing the process using it.
void free_port(int port);

}  // namespace os_util

#include "inline/util/OsUtil.inl"
