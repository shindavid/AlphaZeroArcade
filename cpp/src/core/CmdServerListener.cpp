#include <core/CmdServerListener.hpp>

#include <core/CmdServerClient.hpp>
#include <util/Exception.hpp>

namespace core {

void CmdServerListener::subscribe() {
  if (!CmdServerClient::initialized()) {
    throw util::CleanException("CmdServerClient not initialized. Try passing --cmd-server-port");
  }
  CmdServerClient::get()->add(this);
}

}  // namespace core
