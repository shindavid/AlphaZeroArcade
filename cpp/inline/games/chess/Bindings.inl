#include "games/chess/Bindings.hpp"

namespace a0achess {

inline Keys::EvalKey Keys::eval_key(InputTensorizor* input_tensorizor) {
  return input_tensorizor->current_unit().hash();
}

}  // namespace a0achess
