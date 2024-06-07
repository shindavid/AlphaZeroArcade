// #pragma once

// #include <torch/torch.h>

// #include <core/DerivedTypes.hpp>
// #include <core/GameStateHistory.hpp>
// #include <core/TensorizorConcept.hpp>
// #include <util/CppUtil.hpp>
// #include <util/EigenUtil.hpp>
// #include <util/MetaProgramming.hpp>
// #include <util/TorchUtil.hpp>

// #include <games/connect4/Constants.hpp>
// #include <games/connect4/Game.hpp>

// namespace c4 {

// class OwnershipTarget {
//  public:
//   static constexpr const char* kName = "ownership";
//   using Shape = eigen_util::Shape<kNumRows, kNumColumns>;
//   using Tensor = Eigen::TensorFixedSize<torch_util::dtype, Shape, Eigen::RowMajor>;

//   static void tensorize(Tensor& tensor, const GameState::Data& cur_state,
//                         const GameState::Data& final_state) {
//     core::seat_index_t cp = cur_state.get_current_player();
//     for (int row = 0; row < kNumRows; ++row) {
//       for (int col = 0; col < kNumColumns; ++col) {
//         core::seat_index_t p = final_state.get_player_at(row, col);
//         int val = (p == -1) ? 0 : ((p == cp) ? 2 : 1);
//         tensor(row, col) = val;
//       }
//     }
//   }
// };
// static_assert(core::concepts::AuxTarget<OwnershipTarget, GameState>);

// class Tensorizor {
//  public:
//   static constexpr int kHistorySize = 0;
//   using GameStateHistory = core::GameStateHistory<GameState, kHistorySize>;

//   using InputShape = eigen_util::Shape<kNumPlayers, kNumRows, kNumColumns>;
//   using InputTensor = Eigen::TensorFixedSize<torch_util::dtype, InputShape, Eigen::RowMajor>;

//   using AuxTargetList = mp::TypeList<OwnershipTarget>;

//   static void tensorize(InputTensor& tensor, const GameState::Data& data,
//                         const GameStateHistory& prev_states) {
//     core::seat_index_t cp = data.get_current_player();
//     for (int row = 0; row < kNumRows; ++row) {
//       for (int col = 0; col < kNumColumns; ++col) {
//         core::seat_index_t p = data.get_player_at(row, col);
//         tensor(0, row, col) = (p == cp);
//         tensor(1, row, col) = (p == 1 - cp);
//       }
//     }
//   }
// };

// }  // namespace c4

// static_assert(core::TensorizorConcept<c4::Tensorizor, c4::GameState>);
