#include <connect4/C4Tensorizor.hpp>

#include <torch/torch.h>

namespace c4 {

//template<int tNumPreviousStates>
//inline HistoryBuffer<tNumPreviousStates>::HistoryBuffer()
//  : full_mask_(torch::zeros({kNumPlayers, kHistoryBufferLength, kNumColumns, kNumRows}))
//{
//}

inline void Tensorizor::tensorize(torch::Tensor tensor, const GameState& state) {

}

}  // namespace c4
