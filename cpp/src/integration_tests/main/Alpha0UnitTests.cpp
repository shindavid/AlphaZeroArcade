#include "alpha0/GameLog.hpp"
#include "alpha0/GraphTraits.hpp"
#include "alpha0/Manager.hpp"
#include "alpha0/ManagerParams.hpp"
#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "core/SpecTransforms.hpp"
#include "games/nim/Bindings.hpp"
#include "games/stochastic_nim/Bindings.hpp"
#include "games/stochastic_nim/Constants.hpp"
#include "games/tictactoe/Bindings.hpp"
#include "search/DataLoader.hpp"
#include "search/GameLogCommon.hpp"
#include "search/LookupTable.hpp"
#include "search/NNEvalTraits.hpp"
#include "search/NNEvaluation.hpp"
#include "search/SearchLog.hpp"
#include "search/SearchParams.hpp"
#include "search/SearchRequest.hpp"
#include "search/SimpleNNEvaluationService.hpp"
#include "util/BoostUtil.hpp"
#include "util/GTestUtil.hpp"
#include "util/RepoUtil.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using NimSpec = transforms::AddStateStorage<nim::alpha0::Spec>;
using StochasticNimSpec = transforms::AddStateStorage<stochastic_nim::alpha0::Spec>;
using TicTacToeSpec = transforms::AddStateStorage<tictactoe::alpha0::Spec>;

using NimTraits = NimSpec;
using StochasticNimTraits = StochasticNimSpec;
using TicTacToeTraits = TicTacToeSpec;

template <alpha0::concepts::Spec Spec>
class MockNNEvaluationService
    : public search::SimpleNNEvaluationService<
        search::NNEvalTraits<alpha0::GraphTraits<Spec>, typename Spec::TensorEncodings,
                             search::NNEvaluation<typename Spec::Game, typename Spec::InputFrame,
                                                  typename Spec::NetworkHeads>>> {
 public:
  using Game = Spec::Game;
  using GameTraits = Game::Types;
  using State = Game::State;
  using MoveSet = Game::MoveSet;
  using InputFrame = Spec::InputFrame;
  using NetworkHeads = Spec::NetworkHeads;
  using NNEvaluation = search::NNEvaluation<Game, InputFrame, NetworkHeads>;
  using TensorEncodings = Spec::TensorEncodings;
  using GraphTraits = alpha0::GraphTraits<Spec>;
  using NNEvalTraits = search::NNEvalTraits<GraphTraits, TensorEncodings, NNEvaluation>;
  using Base = search::SimpleNNEvaluationService<NNEvalTraits>;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using GameResultTensor = GameResultEncoding::Tensor;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using Item = Base::Item;

  MockNNEvaluationService(bool smart) : smart_(smart) {
    this->set_init_func([&](NNEvaluation* eval, const Item& item) { this->init_eval(eval, item); });
  }

  void init_eval(NNEvaluation* eval, const Item& item) {
    GameResultTensor value;
    PolicyTensor policy;
    ActionValueTensor action_values;
    group::element_t sym = group::kIdentity;

    core::seat_index_t seat = item.node()->stable_data().active_seat;

    auto encoder = item.input_encoder();
    const State& state = encoder->current_frame();
    action_values.setZero();

    bool winning = state.stones_left % (1 + nim::kMaxStonesToTake) != 0;
    if (winning) {
      int winning_move = state.stones_left % (1 + nim::kMaxStonesToTake) - 1;

      float winning_action_p = smart_ ? 2 : 0;
      float losing_action_p = smart_ ? 0 : 2;

      // these are logits
      float winning_v = smart_ ? 2 : 0;
      float losing_v = smart_ ? 0 : 2;

      value.setValues({winning_v, losing_v});

      policy.setConstant(losing_action_p);
      policy(winning_move) = winning_action_p;

      action_values(winning_move, 0) = winning_v;
    } else {
      value.setZero();
      policy.setZero();
    }

    auto outputs = std::make_tuple(policy, value, action_values);
    MoveSet valid_moves = Game::Rules::analyze(state).valid_moves();
    using InitParams = NNEvaluation::InitParams;
    InitParams init_params{outputs, valid_moves, item.frame(), sym, seat};
    eval->init(init_params);
  }

 private:
  bool smart_;
};

template <alpha0::concepts::Spec Spec>
class ManagerTest : public testing::Test {
 protected:
  using Game = Spec::Game;
  using Manager = alpha0::Manager<Spec>;
  using ManagerParams = alpha0::ManagerParams<Spec>;
  using Node = alpha0::Node<Spec>;
  using Edge = alpha0::Edge<Spec>;
  using Move = Game::Move;
  using LookupTable = search::LookupTable<alpha0::GraphTraits<Spec>>;
  using ValueArray = Game::Types::ValueArray;
  using InputFrame = Spec::InputFrame;
  using NetworkHeads = Spec::NetworkHeads;
  using NNEvaluation = search::NNEvaluation<Game, InputFrame, NetworkHeads>;
  using GraphTraits = alpha0::GraphTraits<Spec>;
  using TensorEncodings = Spec::TensorEncodings;
  using NNEvalTraits = search::NNEvalTraits<GraphTraits, TensorEncodings, NNEvaluation>;
  using Service = search::NNEvaluationServiceBase<NNEvalTraits>;
  using Service_sptr = Service::sptr;
  using State = Game::State;
  using SearchResults = alpha0::SearchResults<Spec>;
  using SearchLog = search::SearchLog<GraphTraits>;

  static_assert(core::kStoreStates<Spec>, "state-storage required for search-log tests");

 public:
  ManagerTest() : manager_params_(create_manager_params()) {}

  ~ManagerTest() override {
    // delete manager_;
  }

  static ManagerParams create_manager_params() {
    ManagerParams params(search::kCompetition);
    params.no_model = true;
    return params;
  }

  void SetUp() override { util::Random::set_seed(0); }

  void init_manager(Service_sptr service = nullptr) {
    core::GameServerBase* server = nullptr;
    manager_ = new Manager(manager_params_, server, service);
    search_log_ = new SearchLog(manager_->lookup_table());
    manager_->set_post_visit_func([&] { search_log_->update(); });
  }

  void start_manager(const std::vector<Move>& initial_moves = {}) {
    manager_->start();
    for (Move move : initial_moves) {
      manager_->update(move);
    }
    this->initial_moves_ = initial_moves;
  }

  ManagerParams& manager_params() { return manager_params_; }

  const SearchResults* search(int num_searches = 0) {
    search::SearchParams search_params(num_searches, true);
    manager_->set_search_params(search_params);
    search::SearchRequest request;
    return manager_->search(request).results;
  }

  Node* get_node_by_index(core::node_pool_index_t index) {
    return manager_->shared_data()->lookup_table.get_node(index);
  }

  SearchLog* get_search_log() { return search_log_; }
  ManagerParams& get_manager_params() { return manager_params_; }

  void test_search(const std::string& testname, int num_search,
                   const std::vector<Move>& initial_moves, Service_sptr service) {
    init_manager(service);
    start_manager(initial_moves);
    const SearchResults* result = search(num_search);

    auto root = util::Repo::root();
    boost::filesystem::path base_dir = root / "goldenfiles" / "alpha0_tests";

    boost::filesystem::path file_path_result = base_dir / (testname + "_result.json");
    boost::filesystem::path file_path_graph = base_dir / (testname + "_graph.json");

    std::stringstream ss_result;
    boost_util::pretty_print(ss_result, result->to_json());

    if (gtest_util::write_goldenfiles) {
      boost_util::write_str_to_file(ss_result.str(), file_path_result);
      boost_util::write_str_to_file(get_search_log()->last_graph_json_str(), file_path_graph);

      boost::filesystem::path log_dir = root / "sample_search_logs" / "alpha0_tests";
      boost::filesystem::path log_file_path = log_dir / (testname + "_log.json");
      boost_util::write_str_to_file(get_search_log()->json_str(), log_file_path);
    }

    std::ifstream result_file(file_path_result);
    std::ifstream graph_file(file_path_graph);

    std::string expected_result_json((std::istreambuf_iterator<char>(result_file)),
                                     std::istreambuf_iterator<char>());
    std::string expected_graph_json((std::istreambuf_iterator<char>(graph_file)),
                                    std::istreambuf_iterator<char>());

    EXPECT_EQ(ss_result.str(), expected_result_json);
    EXPECT_EQ(get_search_log()->last_graph_json_str(), expected_graph_json);
  }

 private:
  ManagerParams manager_params_;
  Manager* manager_ = nullptr;
  std::vector<Move> initial_moves_;
  SearchLog* search_log_ = nullptr;
};

using NimManagerTest = ManagerTest<NimTraits>;
TEST_F(NimManagerTest, uniform_search) {
  std::vector<Move> initial_moves = {nim::kTake3, nim::kTake3, nim::kTake3,
                                     nim::kTake3, nim::kTake3, nim::kTake2};
  test_search("nim_uniform_10", 10, initial_moves, nullptr);
}

TEST_F(NimManagerTest, smart_search) {
  std::shared_ptr<MockNNEvaluationService<NimTraits>> mock_service =
    std::make_shared<MockNNEvaluationService<NimTraits>>(true);

  std::vector<Move> initial_moves = {nim::kTake3, nim::kTake3, nim::kTake3,
                                     nim::kTake3, nim::kTake3, nim::kTake2};
  test_search("nim_smart_service", 10, initial_moves, mock_service);
}

TEST_F(NimManagerTest, dumb_search) {
  std::shared_ptr<MockNNEvaluationService<NimTraits>> mock_service =
    std::make_shared<MockNNEvaluationService<NimTraits>>(false);

  std::vector<Move> initial_moves = {nim::kTake3, nim::kTake3, nim::kTake3,
                                     nim::kTake3, nim::kTake3, nim::kTake2};

  test_search("nim_dumb_service", 10, initial_moves, mock_service);
}

TEST_F(NimManagerTest, 20_searches_from_scratch) { test_search("nim_uniform", 20, {}, nullptr); }

TEST_F(NimManagerTest, 40_searches_from_4_stones) {
  std::vector<nim::Move> initial_moves = {nim::kTake3, nim::kTake3, nim::kTake3,
                                          nim::kTake3, nim::kTake3, nim::kTake2};
  test_search("nim_4_stones", 40, initial_moves, nullptr);
}

TEST_F(NimManagerTest, 40_searches_from_5_stones) {
  std::vector<nim::Move> initial_moves = {nim::kTake3, nim::kTake3, nim::kTake3,
                                          nim::kTake3, nim::kTake3, nim::kTake1};
  test_search("nim_5_stones", 40, initial_moves, nullptr);
}

using StochasticNimManagerTest = ManagerTest<StochasticNimTraits>;
TEST_F(StochasticNimManagerTest, uniform_search) {
  std::vector<stochastic_nim::Move> initial_moves = {
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(2, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(2, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(1, stochastic_nim::kChancePhase),
  };

  test_search("stochastic_nim_uniform_10", 10, initial_moves, nullptr);
}

TEST_F(StochasticNimManagerTest, 20_searches_from_scratch) {
  test_search("stochastic_nim_uniform", 20, {}, nullptr);
}

TEST_F(StochasticNimManagerTest, 100_searches_from_4_stones) {
  std::vector<stochastic_nim::Move> initial_moves = {
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake2, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase)};

  test_search("stochastic_nim_4_stones", 100, initial_moves, nullptr);
}

TEST_F(StochasticNimManagerTest, 100_searches_from_5_stones) {
  std::vector<stochastic_nim::Move> initial_moves = {
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake1, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase)};

  test_search("stochastic_nim_5_stones", 100, initial_moves, nullptr);
}

TEST_F(StochasticNimManagerTest, 100_searches_from_6_stones) {
  std::vector<stochastic_nim::Move> initial_moves = {
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase),
    stochastic_nim::Move(stochastic_nim::kTake3, stochastic_nim::kPlayerPhase),
    stochastic_nim::Move(0, stochastic_nim::kChancePhase)};

  test_search("stochastic_nim_6_stones", 100, initial_moves, nullptr);
}

using TicTacToeManagerTest = ManagerTest<TicTacToeTraits>;
TEST_F(TicTacToeManagerTest, uniform_search_log) {
  std::vector<tictactoe::Move> initial_moves = {0, 1, 2, 4, 7};
  test_search("tictactoe_uniform", 40, initial_moves, nullptr);
}

// ============================================================================
// DataLoader tests (moved from DataLoaderUnitTests.cpp)
// ============================================================================

using NimGame = NimSpec::Game;
using NimState = NimGame::State;
using NimMove = NimGame::Move;
using NimRules = NimGame::Rules;
using NimInputFrame = NimSpec::InputFrame;
using NimInputEncoder = NimSpec::TensorEncodings::InputEncoder;
using NimInputTensor = NimInputEncoder::Tensor;
using NimTensorEncodings = NimSpec::TensorEncodings;
using NimPolicyTensor = NimTensorEncodings::PolicyEncoding::Tensor;
using NimActionValueTensor = NimTensorEncodings::ActionValueEncoding::Tensor;
using NimGameResultTensor = NimTensorEncodings::GameResultEncoding::Tensor;
using NimTrainingInfo = alpha0::TrainingInfo<NimSpec>;
using NimGameWriteLog = alpha0::GameWriteLog<NimSpec>;
using NimGameReadLog = alpha0::GameReadLog<NimSpec>;
using NimTrainingTargets = NimSpec::TrainingTargets::List;

static constexpr int kNimNumTargets = mp::Length_v<NimTrainingTargets>;
static_assert(kNimNumTargets == 4);

static NimTrainingInfo make_nim_training_info(const NimState& state, NimMove move,
                                              float policy_fill, float av_fill,
                                              bool use_for_training) {
  NimTrainingInfo info;
  info.frame = state;
  info.move = move;
  info.active_seat = NimRules::get_current_player(state);
  info.use_for_training = use_for_training;
  info.policy_target.setConstant(policy_fill);
  info.policy_target_valid = true;
  info.action_values_target.setConstant(av_fill);
  info.action_values_target_valid = true;
  return info;
}

static char* nim_assemble_file_buffer(const std::vector<search::GameLogMetadata>& metadata_vec,
                                      const std::vector<char>& data_buf, int total_rows,
                                      int64_t& out_buf_size) {
  int n_games = metadata_vec.size();
  std::vector<char> file_buf;

  search::GameLogFileHeader header;
  header.num_games = n_games;
  header.num_rows = total_rows;
  search::GameLogCommon::write_section(file_buf, &header);

  uint32_t offset_adjustment =
    sizeof(search::GameLogFileHeader) + sizeof(search::GameLogMetadata) * n_games;
  std::vector<search::GameLogMetadata> adjusted = metadata_vec;
  for (auto& md : adjusted) md.start_offset += offset_adjustment;

  search::GameLogCommon::write_section(file_buf, adjusted.data(), n_games);
  file_buf.insert(file_buf.end(), data_buf.begin(), data_buf.end());

  out_buf_size = file_buf.size();
  char* result = new char[out_buf_size];
  std::memcpy(result, file_buf.data(), out_buf_size);
  return result;
}

struct NimGameRecord {
  std::vector<NimState> states;
  std::vector<NimMove> moves;
  NimState final_state;
};

static NimGameRecord play_short_nim_game() {
  NimGameRecord rec;
  NimState state;
  NimRules::init_state(state);
  NimMove moves_seq[] = {NimMove(0), NimMove(1), NimMove(2), NimMove(0), NimMove(1)};
  for (NimMove m : moves_seq) {
    rec.states.push_back(state);
    rec.moves.push_back(m);
    NimRules::apply(state, m);
  }
  rec.final_state = state;
  return rec;
}

struct SerializedNimGame {
  search::GameLogMetadata metadata;
  std::vector<char> data_buf;
  int num_samples;
  NimGameResultTensor outcome;
  std::vector<NimTrainingInfo> training_infos;
};

static SerializedNimGame build_and_serialize_nim_game(const NimGameRecord& game, float policy_base,
                                                      float av_base) {
  SerializedNimGame result;
  NimGameWriteLog write_log(/*id=*/1, /*start_timestamp=*/12345);
  for (int i = 0; i < (int)game.states.size(); ++i) {
    float policy_fill = policy_base + 0.01f * i;
    float av_fill = av_base + 0.01f * i;
    NimTrainingInfo info =
      make_nim_training_info(game.states[i], game.moves[i], policy_fill, av_fill, /*use=*/true);
    write_log.add(info);
    result.training_infos.push_back(info);
  }
  result.outcome.setValues({1.0f, 0.0f});
  write_log.add_terminal(game.final_state, result.outcome);

  search::GameLogSerializer serializer;
  result.metadata = serializer.serialize(&write_log, result.data_buf, /*client_id=*/0);
  result.num_samples = write_log.sample_count();
  return result;
}

static int compute_nim_row_size() {
  int row_size = NimInputTensor::Dimensions::total_size;
  mp::for_each<NimTrainingTargets>(
    [&]<class T>() { row_size += T::Tensor::Dimensions::total_size + 1; });
  return row_size;
}

// Zero out tensor bytes for any target whose mask==0. This makes row comparison independent
// of whatever uninitialized values the encoder leaves behind when encode() returns false.
static void normalize_nim_row(std::vector<float>& row) {
  int offset = NimInputTensor::Dimensions::total_size;
  mp::for_each<NimTrainingTargets>([&]<class T>() {
    constexpr int tensor_size = T::Tensor::Dimensions::total_size;
    if (row[offset + tensor_size] == 0.0f) {
      std::fill(row.begin() + offset, row.begin() + offset + tensor_size, 0.0f);
    }
    offset += tensor_size + 1;
  });
}

TEST(NimGameLogRoundTrip, SerializeDeserialize) {
  NimGameRecord game = play_short_nim_game();
  SerializedNimGame sg = build_and_serialize_nim_game(game, /*policy_base=*/0.5f, /*av_base=*/0.1f);

  std::vector<search::GameLogMetadata> metadata_vec = {sg.metadata};
  int64_t buf_size;
  char* buf = nim_assemble_file_buffer(metadata_vec, sg.data_buf, sg.num_samples, buf_size);

  search::GameLogFileReader reader(buf);
  ASSERT_EQ(reader.num_games(), 1);
  ASSERT_EQ(reader.num_samples(0), sg.num_samples);

  std::vector<int> target_indices = {0, 1, 2, 3};
  int row_size = compute_nim_row_size();

  NimGameReadLog read_log("test", 0, reader.metadata(0), reader.game_data_buffer(0));
  ASSERT_EQ(read_log.num_sampled_frames(), sg.num_samples);

  for (int row = 0; row < sg.num_samples; ++row) {
    std::vector<float> output(row_size, -999.0f);
    read_log.load(row, /*apply_symmetry=*/false, target_indices, output.data());

    const NimTrainingInfo& expected_info = sg.training_infos[row];
    NimInputEncoder encoder;
    encoder.restore(&expected_info.frame, 1);
    NimInputTensor expected_input = encoder.encode();

    constexpr int kInputSize = NimInputTensor::Dimensions::total_size;
    for (int i = 0; i < kInputSize; ++i) {
      EXPECT_FLOAT_EQ(output[i], expected_input.data()[i])
        << "Input mismatch at row=" << row << " i=" << i;
    }

    int offset = kInputSize;

    // Policy target
    for (int i = 0; i < NimPolicyTensor::Dimensions::total_size; ++i) {
      EXPECT_FLOAT_EQ(output[offset + i], expected_info.policy_target.data()[i])
        << "Policy mismatch at row=" << row << " i=" << i;
    }
    offset += NimPolicyTensor::Dimensions::total_size;
    EXPECT_FLOAT_EQ(output[offset], 1.0f);
    offset++;

    // Value target: left-rotate by active_seat
    NimGameResultTensor expected_value = sg.outcome;
    NimTensorEncodings::GameResultEncoding::left_rotate(expected_value, expected_info.active_seat);
    for (int i = 0; i < NimGameResultTensor::Dimensions::total_size; ++i) {
      EXPECT_FLOAT_EQ(output[offset + i], expected_value.data()[i])
        << "Value mismatch at row=" << row << " i=" << i;
    }
    offset += NimGameResultTensor::Dimensions::total_size;
    EXPECT_FLOAT_EQ(output[offset], 1.0f);
    offset++;

    // ActionValue target: left-rotate by active_seat
    NimActionValueTensor expected_av = expected_info.action_values_target;
    eigen_util::left_rotate(expected_av, expected_info.active_seat);
    for (int i = 0; i < NimActionValueTensor::Dimensions::total_size; ++i) {
      EXPECT_FLOAT_EQ(output[offset + i], expected_av.data()[i])
        << "ActionValue mismatch at row=" << row << " i=" << i;
    }
    offset += NimActionValueTensor::Dimensions::total_size;
    EXPECT_FLOAT_EQ(output[offset], 1.0f);
    offset++;

    // OppPolicy target
    if (row + 1 < (int)sg.training_infos.size()) {
      for (int i = 0; i < NimPolicyTensor::Dimensions::total_size; ++i) {
        EXPECT_FLOAT_EQ(output[offset + i], sg.training_infos[row + 1].policy_target.data()[i])
          << "OppPolicy mismatch at row=" << row << " i=" << i;
      }
      offset += NimPolicyTensor::Dimensions::total_size;
      EXPECT_FLOAT_EQ(output[offset], 1.0f);
    } else {
      offset += NimPolicyTensor::Dimensions::total_size;
      EXPECT_FLOAT_EQ(output[offset], 0.0f);
    }
    offset++;

    EXPECT_EQ(offset, row_size);
  }

  delete[] buf;
}

struct NimDataLoaderConfig {
  int num_workers;
  int num_prefetch;
  int64_t memory_budget;
};

class NimDataLoaderRoundTrip : public ::testing::TestWithParam<NimDataLoaderConfig> {};

TEST_P(NimDataLoaderRoundTrip, LoadMatchesSerializedData) {
  const NimDataLoaderConfig& cfg = GetParam();

  NimGameRecord game1 = play_short_nim_game();
  NimGameRecord game2 = play_short_nim_game();

  SerializedNimGame sg1 =
    build_and_serialize_nim_game(game1, /*policy_base=*/0.5f, /*av_base=*/0.1f);
  SerializedNimGame sg2 =
    build_and_serialize_nim_game(game2, /*policy_base=*/0.8f, /*av_base=*/0.3f);

  int total_rows = sg1.num_samples + sg2.num_samples;

  std::vector<search::GameLogMetadata> metadata_vec = {sg1.metadata, sg2.metadata};
  std::vector<char> combined_data = sg1.data_buf;
  search::GameLogMetadata md2_adjusted = sg2.metadata;
  md2_adjusted.start_offset += sg1.data_buf.size();
  metadata_vec[1] = md2_adjusted;
  combined_data.insert(combined_data.end(), sg2.data_buf.begin(), sg2.data_buf.end());

  int64_t buf_size;
  char* buf = nim_assemble_file_buffer(metadata_vec, combined_data, total_rows, buf_size);

  search::DataLoaderBase::Params params{"unused", cfg.memory_budget, cfg.num_workers,
                                        cfg.num_prefetch};
  search::DataLoader<NimGameReadLog> loader(params);
  loader.test_add_gen_from_buffer(/*gen=*/1, total_rows, buf, buf_size);

  std::vector<int> target_indices = {0, 1, 2, 3};
  int row_size = compute_nim_row_size();
  std::vector<float> output(total_rows * row_size, -999.0f);
  int gen_range[2] = {0, 0};
  int version_check[2] = {0, 0};

  search::DataLoaderBase::LoadParams load_params;
  load_params.window_start = 0;
  load_params.window_end = total_rows;
  load_params.n_samples = total_rows;
  load_params.apply_symmetry = false;
  load_params.n_targets = kNimNumTargets;
  load_params.output_array = output.data();
  load_params.target_indices_array = target_indices.data();
  load_params.gen_range = gen_range;
  load_params.version_check = version_check;
  loader.load(load_params);

  EXPECT_EQ(version_check[0], 0);
  EXPECT_EQ(version_check[1], 0);

  char* ref_buf = new char[buf_size];
  std::memcpy(ref_buf, buf, buf_size);

  search::GameLogFileReader reader(ref_buf);
  ASSERT_EQ(reader.num_games(), 2);

  std::vector<std::vector<float>> expected_rows;
  for (int g = 0; g < reader.num_games(); ++g) {
    NimGameReadLog read_log("ref", g, reader.metadata(g), reader.game_data_buffer(g));
    for (int r = 0; r < read_log.num_sampled_frames(); ++r) {
      std::vector<float> row(row_size);
      read_log.load(r, false, target_indices, row.data());
      expected_rows.push_back(row);
    }
  }
  delete[] ref_buf;

  ASSERT_EQ((int)expected_rows.size(), total_rows);

  for (auto& row : expected_rows) normalize_nim_row(row);

  for (int i = 0; i < total_rows; ++i) {
    std::vector<float> actual_row(output.begin() + i * row_size,
                                  output.begin() + (i + 1) * row_size);
    normalize_nim_row(actual_row);
    bool found = false;
    for (const auto& expected_row : expected_rows) {
      if (actual_row == expected_row) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Row " << i << " from DataLoader does not match any expected row";
  }
}

INSTANTIATE_TEST_SUITE_P(
  Configs, NimDataLoaderRoundTrip,
  ::testing::Values(NimDataLoaderConfig{1, 0, 1 << 30}, NimDataLoaderConfig{2, 0, 1 << 30},
                    NimDataLoaderConfig{4, 0, 1 << 30}, NimDataLoaderConfig{1, 1, 1 << 30},
                    NimDataLoaderConfig{2, 2, 1 << 30}, NimDataLoaderConfig{1, 0, 256}),
  [](const ::testing::TestParamInfo<NimDataLoaderConfig>& p) {
    return std::format("w{}_p{}_m{}", p.param.num_workers, p.param.num_prefetch,
                       p.param.memory_budget);
  });

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
