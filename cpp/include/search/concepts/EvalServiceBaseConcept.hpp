#pragma once

#include "core/BasicTypes.hpp"
#include "search/NNEvaluationRequest.hpp"

#include <concepts>

namespace search {
namespace concepts {

template <class E, class Traits>
concept EvalServiceBase = requires(E& service, search::NNEvaluationRequest<Traits>& request) {
  // An EvalServiceBase must have a connect() method, which will be called in
  // search::Manager::start(). This is the appropriate place to start any threads used to support
  // the key evaluate() method.
  { service.connect() };

  // An EvalServiceBase must have a disconnect() method, which will be called in the search::Manager
  // destructor. This is the appropriate place to clean up any resources instantiated inside
  // connect().
  { service.disconnect() };

  // An EvalServiceBase must have an end_session() method, which will be called during program
  // shutdown. Compared to disconnect(), you can have more certainty about the state of the
  // program when this is called. We currently use this to collect certain profiling stats, which
  // is broadcasted to an external server. Doing this in the destructor would require extra
  // sequencing constraints (for example, ensuring that the connection to the external server is
  // still around). We make a more explicit spot to put actions like this, which helps with
  // clarity and maintainability.
  { service.end_session() };

  // The key method of EvalServiceBase. This method is responsible for processing the evaluation
  // request.
  //
  // If the request will be processed asynchronously, should return core::kYield, and then notify
  // request.notification_unit() when the evaluation is complete. Else, should return
  // core::kContinue.
  //
  // Completing the evaluation entails call item.set_eval() for each item in request.fresh_items().
  { service.evaluate(request) } -> std::same_as<core::yield_instruction_t>;
};

}  // namespace concepts
}  // namespace search
