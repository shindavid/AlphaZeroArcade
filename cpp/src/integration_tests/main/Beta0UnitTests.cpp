#include "beta0/GameLog.hpp"
#include "beta0/GraphTraits.hpp"
#include "beta0/Manager.hpp"
#include "beta0/ManagerParams.hpp"
#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "core/SpecTransforms.hpp"
#include "games/connect4/Bindings.hpp"
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

using C4Spec = transforms::AddStateStorage<c4::beta0::Spec>;

template <beta0::concepts::Spec Spec>
class MockNNEvaluationService
    : public search::SimpleNNEvaluationService<
        search::NNEvalTraits<beta0::GraphTraits<Spec>, typename Spec::TensorEncodings,
                             search::NNEvaluation<typename Spec::Game, typename Spec::InputFrame,
                                                  typename Spec::NetworkHeads>>> {
 public:
  using Game = Spec::Game;
  using State = Game::State;
  using MoveSet = Game::MoveSet;
  using InputFrame = Spec::InputFrame;
  using NetworkHeads = Spec::NetworkHeads;
  using NNEvaluation = search::NNEvaluation<Game, InputFrame, NetworkHeads>;
  using TensorEncodings = Spec::TensorEncodings;
  using NNEvalTraits =
    search::NNEvalTraits<beta0::GraphTraits<Spec>, TensorEncodings, NNEvaluation>;
  using Base = search::SimpleNNEvaluationService<NNEvalTraits>;
  using GameResultEncoding = TensorEncodings::GameResultEncoding;
  using GameResultTensor = GameResultEncoding::Tensor;
  using WinShareTensor = TensorEncodings::WinShareTensor;
  using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
  using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
  using NetworkHeadsList = NetworkHeads::List;
  using BackupAccuStaticHead = mp::TypeAt_t<NetworkHeadsList, 5>;
  using BackupAccuStaticTensor = BackupAccuStaticHead::Tensor;
  using Item = Base::Item;

  MockNNEvaluationService() {
    this->set_init_func([&](NNEvaluation* eval, const Item& item) { this->init_eval(eval, item); });
  }

  void init_eval(NNEvaluation* eval, const Item& item) {
    group::element_t sym = group::kIdentity;
    core::seat_index_t seat = item.node()->stable_data().active_seat;

    const State* state_ptr = item.node()->stable_data().get_state();
    RELEASE_ASSERT(state_ptr != nullptr, "state storage must be enabled for beta0 tests");
    MoveSet valid_moves = Game::Rules::analyze(*state_ptr).valid_moves();

    GameResultTensor value;
    value.setZero();  // logits = 0 → neutral evaluation

    PolicyTensor policy;
    policy.setZero();  // logits = 0 → uniform after softmax

    WinShareTensor uncertainty;
    uncertainty.setConstant(0.1f);  // small constant prior uncertainty

    ActionValueTensor action_values;
    action_values.setZero();

    ActionValueTensor action_values_uncertainty;
    action_values_uncertainty.setConstant(0.1f);

    BackupAccuStaticTensor backup_accu_static;
    backup_accu_static.setZero();  // static accumulator portion = 0 (GPU contribution)

    auto outputs = std::make_tuple(policy, value, uncertainty, action_values,
                                   action_values_uncertainty, backup_accu_static);
    using InitParams = NNEvaluation::InitParams;
    InitParams init_params{outputs, valid_moves, item.frame(), sym, seat};
    eval->init(init_params);
  }
};

template <beta0::concepts::Spec Spec>
class ManagerTest : public testing::Test {
 protected:
  using Game = Spec::Game;
  using Manager = beta0::Manager<Spec>;
  using ManagerParams = beta0::ManagerParams<Spec>;
  using Node = beta0::Node<Spec>;
  using Edge = beta0::Edge<Spec>;
  using Move = Game::Move;
  using LookupTable = search::LookupTable<beta0::GraphTraits<Spec>>;
  using InputFrame = Spec::InputFrame;
  using NetworkHeads = Spec::NetworkHeads;
  using NNEvaluation = search::NNEvaluation<Game, InputFrame, NetworkHeads>;
  using NNEvalTraits =
    search::NNEvalTraits<beta0::GraphTraits<Spec>, typename Spec::TensorEncodings, NNEvaluation>;
  using Service = search::NNEvaluationServiceBase<NNEvalTraits>;
  using Service_sptr = Service::sptr;
  using SearchResults = beta0::SearchResults<Spec>;
  using SearchLog = search::SearchLog<beta0::GraphTraits<Spec>>;
  using MockService = MockNNEvaluationService<Spec>;

  static_assert(core::kStoreStates<Spec>, "state-storage required for search-log tests");

 public:
  ManagerTest() : manager_params_(create_manager_params()) {}

  ~ManagerTest() override {
    delete manager_;
    delete search_log_;
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
  }

  ManagerParams& manager_params() { return manager_params_; }

  const SearchResults* search(int num_searches = 0) {
    search::SearchParams search_params(num_searches, true);
    manager_->set_search_params(search_params);
    search::SearchRequest request;
    return manager_->search(request).results;
  }

  SearchLog* get_search_log() { return search_log_; }

  void test_search(const std::string& testname, int num_search,
                   const std::vector<Move>& initial_moves, Service_sptr service,
                   const std::vector<float>& backup_weights = {}) {
    init_manager(service);
    if (!backup_weights.empty()) {
      manager_->set_backup_nn_weights(backup_weights.data(), backup_weights.size());
    }
    start_manager(initial_moves);
    const SearchResults* result = search(num_search);

    auto root = util::Repo::root();
    boost::filesystem::path base_dir = root / "goldenfiles" / "beta0_tests";

    boost::filesystem::path file_path_result = base_dir / (testname + "_result.json");
    boost::filesystem::path file_path_graph = base_dir / (testname + "_graph.json");

    std::stringstream ss_result;
    boost_util::pretty_print(ss_result, result->to_json());

    if (gtest_util::write_goldenfiles) {
      boost_util::write_str_to_file(ss_result.str(), file_path_result);
      boost_util::write_str_to_file(get_search_log()->last_graph_json_str(), file_path_graph);

      boost::filesystem::path log_dir = root / "sample_search_logs" / "beta0_tests";
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
  SearchLog* search_log_ = nullptr;
};

using C4ManagerTest = ManagerTest<C4Spec>;

/*
 * Test 1: BetaZero MCTS with no backup NN loaded — uses the LoTV (Law of Total Variance)
 * path for W estimation.
 */
TEST_F(C4ManagerTest, no_backup_nn) {
  auto service = std::make_shared<MockNNEvaluationService<C4Spec>>();
  test_search("c4_no_backup_nn", 10, {}, service);
}

/*
 * Test 2: BetaZero MCTS with backup NN weights loaded.
 * W_child = 0, W_out = 0, b_out = [0.5, 0.5, 0.1, 0.1].
 * The network override produces Q=[0.5,0.5], W=[0.1,0.1] for all nodes.
 *
 * For c4 beta0: kChildInputDim=5, kBackupHiddenDim=64, kOutputDim=4
 * Weight layout:
 *   W_child [5 * 64 = 320 floats]  → all zero
 *   W_out   [64 * 4 = 256 floats]  → all zero
 *   b_out   [4 floats]             → [0.5, 0.5, 0.1, 0.1]
 * Total: 580 floats
 */
TEST_F(C4ManagerTest, with_backup_nn) {
  using BackupNNEvaluator = beta0::BackupNNEvaluator<C4Spec>;
  constexpr size_t kWeightCount = BackupNNEvaluator::kWeightCount;  // 580 for c4
  constexpr int kOutputDim = BackupNNEvaluator::kOutputDim;         // 4 for c4

  std::vector<float> weights(kWeightCount, 0.0f);
  // Set b_out = [0.5, 0.5, 0.1, 0.1]  (last kOutputDim floats)
  weights[kWeightCount - kOutputDim + 0] = 0.5f;
  weights[kWeightCount - kOutputDim + 1] = 0.5f;
  weights[kWeightCount - kOutputDim + 2] = 0.1f;
  weights[kWeightCount - kOutputDim + 3] = 0.1f;

  auto service = std::make_shared<MockNNEvaluationService<C4Spec>>();
  test_search("c4_with_backup_nn", 10, {}, service, weights);
}

// ============================================================================
// DataLoader tests (beta0 / c4 versions)
// ============================================================================

using C4Game = C4Spec::Game;
using C4State = C4Game::State;
using C4Rules = C4Game::Rules;
using C4TensorEncodings = C4Spec::TensorEncodings;
using C4InputEncoder = C4TensorEncodings::InputEncoder;
using C4InputTensor = C4InputEncoder::Tensor;
using C4PolicyTensor = C4TensorEncodings::PolicyEncoding::Tensor;
using C4ActionValueTensor = C4TensorEncodings::ActionValueEncoding::Tensor;
using C4GameResultTensor = C4TensorEncodings::GameResultEncoding::Tensor;
using C4WinShareTensor = C4TensorEncodings::WinShareTensor;
using C4ValueArray = C4Game::Types::ValueArray;
using C4TrainingInfo = beta0::TrainingInfo<C4Spec>;
using C4GameWriteLog = beta0::GameWriteLog<C4Spec>;
using C4GameReadLog = beta0::GameReadLog<C4Spec>;
using C4TrainingTargets = C4Spec::TrainingTargets::List;

static constexpr int kC4NumTargets = mp::Length_v<C4TrainingTargets>;
static_assert(kC4NumTargets == 6);

static C4TrainingInfo make_c4_training_info(const C4State& state, c4::Move move, float policy_fill,
                                            float av_fill, float au_fill, bool use_for_training) {
  C4TrainingInfo info;
  info.frame = state;  // c4::State inherits c4::InputFrame
  info.move = move;
  info.active_seat = C4Rules::get_current_player(state);
  info.use_for_training = use_for_training;
  info.policy_target.setConstant(policy_fill);
  info.policy_target_valid = true;
  info.action_values_target.setConstant(av_fill);
  info.action_values_uncertainty_target.setConstant(au_fill);
  info.action_values_target_valid = true;
  // Q_root used by LoTV backward pass in add_terminal() to compute W_target
  info.Q_root[0] = av_fill;
  info.Q_root[1] = 1.0f - av_fill;
  return info;
}

static char* c4_assemble_file_buffer(const std::vector<search::GameLogMetadata>& metadata_vec,
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

struct C4GameRecord {
  std::vector<C4State> states;
  std::vector<c4::Move> moves;
  C4State final_state;
};

static C4GameRecord play_short_c4_game() {
  C4GameRecord rec;
  C4State state;
  C4Rules::init_state(state);
  // 5 moves that don't produce a win: cols 0,1,0,1,2
  // Player 0 drops at col 0, then col 0 again, then col 2
  // Player 1 drops at col 1, then col 1 again
  c4::Move moves_seq[] = {c4::Move(0), c4::Move(1), c4::Move(0), c4::Move(1), c4::Move(2)};
  for (c4::Move m : moves_seq) {
    rec.states.push_back(state);
    rec.moves.push_back(m);
    C4Rules::apply(state, m);
  }
  rec.final_state = state;
  return rec;
}

struct SerializedC4Game {
  search::GameLogMetadata metadata;
  std::vector<char> data_buf;
  int num_samples;
  C4GameResultTensor outcome;
  std::vector<C4TrainingInfo> training_infos;
  std::vector<C4ValueArray> w_targets;  // retroactively computed by add_terminal()
};

static SerializedC4Game build_and_serialize_c4_game(const C4GameRecord& game, float policy_base,
                                                    float av_base) {
  SerializedC4Game result;
  C4GameWriteLog write_log(/*id=*/1, /*start_timestamp=*/12345);
  for (int i = 0; i < (int)game.states.size(); ++i) {
    float policy_fill = policy_base + 0.01f * i;
    float av_fill = av_base + 0.01f * i;
    float au_fill = av_fill * 0.5f;
    C4TrainingInfo info = make_c4_training_info(game.states[i], game.moves[i], policy_fill, av_fill,
                                                au_fill, /*use=*/true);
    write_log.add(info);
    result.training_infos.push_back(info);
  }

  result.outcome.setValues({1.0f, 0.0f, 0.0f});  // player 0 wins (Win, Loss, Draw)
  write_log.add_terminal(game.final_state, result.outcome);

  // Capture retroactively computed W_targets before serializing
  for (size_t i = 0; i < write_log.size(); ++i) {
    result.w_targets.push_back(write_log.get_full_record(i)->W_target);
  }

  search::GameLogSerializer serializer;
  result.metadata = serializer.serialize(&write_log, result.data_buf, /*client_id=*/0);
  result.num_samples = write_log.sample_count();
  return result;
}

static int compute_c4_row_size() {
  int row_size = C4InputTensor::Dimensions::total_size;
  mp::for_each<C4TrainingTargets>([&]<class T>() {
    row_size += T::Tensor::Dimensions::total_size + 1;
  });
  return row_size;
}

// Zero out tensor bytes for any target whose mask==0. This makes row comparison independent
// of whatever uninitialized values the encoder leaves behind when encode() returns false.
static void normalize_c4_row(std::vector<float>& row) {
  int offset = C4InputTensor::Dimensions::total_size;
  mp::for_each<C4TrainingTargets>([&]<class T>() {
    constexpr int tensor_size = T::Tensor::Dimensions::total_size;
    if (row[offset + tensor_size] == 0.0f) {
      std::fill(row.begin() + offset, row.begin() + offset + tensor_size, 0.0f);
    }
    offset += tensor_size + 1;
  });
}

TEST(C4GameLogRoundTrip, SerializeDeserialize) {
  C4GameRecord game = play_short_c4_game();
  SerializedC4Game sg = build_and_serialize_c4_game(game, /*policy_base=*/0.5f, /*av_base=*/0.1f);

  std::vector<search::GameLogMetadata> metadata_vec = {sg.metadata};
  int64_t buf_size;
  char* buf = c4_assemble_file_buffer(metadata_vec, sg.data_buf, sg.num_samples, buf_size);

  search::GameLogFileReader reader(buf);
  ASSERT_EQ(reader.num_games(), 1);
  ASSERT_EQ(reader.num_samples(0), sg.num_samples);

  std::vector<int> target_indices = {0, 1, 2, 3, 4, 5};
  int row_size = compute_c4_row_size();

  C4GameReadLog read_log("test", 0, reader.metadata(0), reader.game_data_buffer(0));
  ASSERT_EQ(read_log.num_sampled_frames(), sg.num_samples);

  for (int row = 0; row < sg.num_samples; ++row) {
    std::vector<float> output(row_size, -999.0f);
    read_log.load(row, /*apply_symmetry=*/false, target_indices, output.data());

    const C4TrainingInfo& expected_info = sg.training_infos[row];
    C4InputEncoder encoder;
    encoder.restore(&expected_info.frame, 1);
    C4InputTensor expected_input = encoder.encode();

    constexpr int kInputSize = C4InputTensor::Dimensions::total_size;
    for (int i = 0; i < kInputSize; ++i) {
      EXPECT_FLOAT_EQ(output[i], expected_input.data()[i])
        << "Input mismatch at row=" << row << " i=" << i;
    }

    int offset = kInputSize;

    // Policy target
    for (int i = 0; i < C4PolicyTensor::Dimensions::total_size; ++i) {
      EXPECT_FLOAT_EQ(output[offset + i], expected_info.policy_target.data()[i])
        << "Policy mismatch at row=" << row << " i=" << i;
    }
    offset += C4PolicyTensor::Dimensions::total_size;
    EXPECT_FLOAT_EQ(output[offset], 1.0f);
    offset++;

    // Value target: left-rotate outcome by active_seat
    C4GameResultTensor expected_value = sg.outcome;
    C4TensorEncodings::GameResultEncoding::left_rotate(expected_value, expected_info.active_seat);
    for (int i = 0; i < C4GameResultTensor::Dimensions::total_size; ++i) {
      EXPECT_FLOAT_EQ(output[offset + i], expected_value.data()[i])
        << "Value mismatch at row=" << row << " i=" << i;
    }
    offset += C4GameResultTensor::Dimensions::total_size;
    EXPECT_FLOAT_EQ(output[offset], 1.0f);
    offset++;

    // ActionValue target: left-rotate by active_seat
    C4ActionValueTensor expected_av = expected_info.action_values_target;
    eigen_util::left_rotate(expected_av, expected_info.active_seat);
    for (int i = 0; i < C4ActionValueTensor::Dimensions::total_size; ++i) {
      EXPECT_FLOAT_EQ(output[offset + i], expected_av.data()[i])
        << "ActionValue mismatch at row=" << row << " i=" << i;
    }
    offset += C4ActionValueTensor::Dimensions::total_size;
    EXPECT_FLOAT_EQ(output[offset], 1.0f);
    offset++;

    // W target: retroactively computed, left-rotated by active_seat
    {
      C4WinShareTensor expected_W;
      for (int p = 0; p < C4Game::Constants::kNumPlayers; ++p) {
        expected_W.data()[p] = sg.w_targets[row][p];
      }
      eigen_util::left_rotate(expected_W, expected_info.active_seat);
      for (int i = 0; i < C4WinShareTensor::Dimensions::total_size; ++i) {
        EXPECT_FLOAT_EQ(output[offset + i], expected_W.data()[i])
          << "W mismatch at row=" << row << " i=" << i;
      }
      offset += C4WinShareTensor::Dimensions::total_size;
      EXPECT_FLOAT_EQ(output[offset], 1.0f);  // W_target_valid = true
      offset++;
    }

    // AU target: left-rotate by active_seat
    C4ActionValueTensor expected_au = expected_info.action_values_uncertainty_target;
    eigen_util::left_rotate(expected_au, expected_info.active_seat);
    for (int i = 0; i < C4ActionValueTensor::Dimensions::total_size; ++i) {
      EXPECT_FLOAT_EQ(output[offset + i], expected_au.data()[i])
        << "AU mismatch at row=" << row << " i=" << i;
    }
    offset += C4ActionValueTensor::Dimensions::total_size;
    EXPECT_FLOAT_EQ(output[offset], 1.0f);
    offset++;

    // OppPolicy target
    if (row + 1 < (int)sg.training_infos.size()) {
      for (int i = 0; i < C4PolicyTensor::Dimensions::total_size; ++i) {
        EXPECT_FLOAT_EQ(output[offset + i], sg.training_infos[row + 1].policy_target.data()[i])
          << "OppPolicy mismatch at row=" << row << " i=" << i;
      }
      offset += C4PolicyTensor::Dimensions::total_size;
      EXPECT_FLOAT_EQ(output[offset], 1.0f);
    } else {
      offset += C4PolicyTensor::Dimensions::total_size;
      EXPECT_FLOAT_EQ(output[offset], 0.0f);
    }
    offset++;

    EXPECT_EQ(offset, row_size);
  }

  delete[] buf;
}

struct C4DataLoaderConfig {
  int num_workers;
  int num_prefetch;
  int64_t memory_budget;
};

class C4DataLoaderRoundTrip : public ::testing::TestWithParam<C4DataLoaderConfig> {};

TEST_P(C4DataLoaderRoundTrip, LoadMatchesSerializedData) {
  const C4DataLoaderConfig& cfg = GetParam();

  C4GameRecord game1 = play_short_c4_game();
  C4GameRecord game2 = play_short_c4_game();

  SerializedC4Game sg1 = build_and_serialize_c4_game(game1, /*policy_base=*/0.5f, /*av_base=*/0.1f);
  SerializedC4Game sg2 = build_and_serialize_c4_game(game2, /*policy_base=*/0.8f, /*av_base=*/0.3f);

  int total_rows = sg1.num_samples + sg2.num_samples;

  std::vector<search::GameLogMetadata> metadata_vec = {sg1.metadata, sg2.metadata};
  std::vector<char> combined_data = sg1.data_buf;
  search::GameLogMetadata md2_adjusted = sg2.metadata;
  md2_adjusted.start_offset += sg1.data_buf.size();
  metadata_vec[1] = md2_adjusted;
  combined_data.insert(combined_data.end(), sg2.data_buf.begin(), sg2.data_buf.end());

  int64_t buf_size;
  char* buf = c4_assemble_file_buffer(metadata_vec, combined_data, total_rows, buf_size);

  search::DataLoaderBase::Params params{"unused", cfg.memory_budget, cfg.num_workers,
                                        cfg.num_prefetch};
  search::DataLoader<C4GameReadLog> loader(params);
  loader.test_add_gen_from_buffer(/*gen=*/1, total_rows, buf, buf_size);

  std::vector<int> target_indices = {0, 1, 2, 3, 4, 5};
  int row_size = compute_c4_row_size();
  std::vector<float> output(total_rows * row_size, -999.0f);
  int gen_range[2] = {0, 0};
  int version_check[2] = {0, 0};

  search::DataLoaderBase::LoadParams load_params;
  load_params.window_start = 0;
  load_params.window_end = total_rows;
  load_params.n_samples = total_rows;
  load_params.apply_symmetry = false;
  load_params.n_targets = kC4NumTargets;
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
    C4GameReadLog read_log("ref", g, reader.metadata(g), reader.game_data_buffer(g));
    for (int r = 0; r < read_log.num_sampled_frames(); ++r) {
      std::vector<float> row(row_size);
      read_log.load(r, false, target_indices, row.data());
      expected_rows.push_back(row);
    }
  }
  delete[] ref_buf;

  ASSERT_EQ((int)expected_rows.size(), total_rows);

  for (auto& row : expected_rows) normalize_c4_row(row);

  for (int i = 0; i < total_rows; ++i) {
    std::vector<float> actual_row(output.begin() + i * row_size,
                                  output.begin() + (i + 1) * row_size);
    normalize_c4_row(actual_row);
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
  Configs, C4DataLoaderRoundTrip,
  ::testing::Values(C4DataLoaderConfig{1, 0, 1 << 30}, C4DataLoaderConfig{2, 0, 1 << 30},
                    C4DataLoaderConfig{4, 0, 1 << 30}, C4DataLoaderConfig{1, 1, 1 << 30},
                    C4DataLoaderConfig{2, 2, 1 << 30}, C4DataLoaderConfig{1, 0, 256}),
  [](const ::testing::TestParamInfo<C4DataLoaderConfig>& p) {
    return std::format("w{}_p{}_m{}", p.param.num_workers, p.param.num_prefetch,
                       p.param.memory_budget);
  });

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
