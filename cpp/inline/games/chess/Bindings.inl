#include "games/chess/Bindings.hpp"

namespace chess {

inline Keys::EvalKey Keys::eval_key(InputTensorizor* input_tensorizor) {
  return input_tensorizor->current_state().history_hash;
}

}  // namespace chess
