#include "alpha0/GameLog.hpp"
#include "core/SpecTransforms.hpp"
#include "games/nim/Bindings.hpp"
#include "search/DataLoader.hpp"
#include "search/GameLogCommon.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

using NimSpec = transforms::AddStateStorage<nim::alpha0::Spec>;

using Game = NimSpec::Game;
using State = Game::State;
using Move = Game::Move;
using Rules = Game::Rules;
using InputFrame = NimSpec::InputFrame;
using InputEncoder = NimSpec::TensorEncodings::InputEncoder;
using InputTensor = InputEncoder::Tensor;
using TensorEncodings = NimSpec::TensorEncodings;
using PolicyTensor = TensorEncodings::PolicyEncoding::Tensor;
using ActionValueTensor = TensorEncodings::ActionValueEncoding::Tensor;
using GameResultTensor = TensorEncodings::GameResultEncoding::Tensor;
using TrainingInfo = alpha0::TrainingInfo<NimSpec>;
using GameWriteLog = alpha0::GameWriteLog<NimSpec>;
using GameReadLog = alpha0::GameReadLog<NimSpec>;
using GameLogSerializer = alpha0::GameLogSerializer<NimSpec>;
using GameLogFileReader = search::GameLogFileReader;
using GameLogFileHeader = search::GameLogFileHeader;
using GameLogMetadata = search::GameLogMetadata;
using GameLogCommon = search::GameLogCommon;
using TrainingTargets = NimSpec::TrainingTargets::List;

// Number of training targets for Nim's StandardTrainingTargets:
//   [PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget]
static constexpr int kNumTargets = mp::Length_v<TrainingTargets>;
static_assert(kNumTargets == 4);

// Helper: build a TrainingInfo with known values for a given game state.
TrainingInfo make_training_info(const State& state, Move move, float policy_fill,
                                float av_fill, bool use_for_training) {
  TrainingInfo info;
  info.frame = state;
  info.move = move;
  info.active_seat = Rules::get_current_player(state);
  info.use_for_training = use_for_training;

  info.policy_target.setConstant(policy_fill);
  info.policy_target_valid = true;

  info.action_values_target.setConstant(av_fill);
  info.action_values_target_valid = true;

  return info;
}

// Helper: assemble a game-log file buffer from serialized game data.
// Returns a new[]-allocated buffer that represents a complete game-log file.
// Also returns total_rows via out-param.
char* assemble_file_buffer(const std::vector<GameLogMetadata>& metadata_vec,
                           const std::vector<char>& data_buf, int total_rows,
                           int64_t& out_buf_size) {
  int n_games = metadata_vec.size();

  // Build the file: [Header][Metadata...][Data...]
  std::vector<char> file_buf;

  GameLogFileHeader header;
  header.num_games = n_games;
  header.num_rows = total_rows;

  GameLogCommon::write_section(file_buf, &header);

  // We need to adjust start_offsets: currently relative to data_buf start,
  // but need to be relative to file start.
  uint32_t offset_adjustment = sizeof(GameLogFileHeader) + sizeof(GameLogMetadata) * n_games;
  std::vector<GameLogMetadata> adjusted = metadata_vec;
  for (auto& md : adjusted) {
    md.start_offset += offset_adjustment;
  }

  GameLogCommon::write_section(file_buf, adjusted.data(), n_games);
  file_buf.insert(file_buf.end(), data_buf.begin(), data_buf.end());

  out_buf_size = file_buf.size();
  char* result = new char[out_buf_size];
  std::memcpy(result, file_buf.data(), out_buf_size);
  return result;
}

// Play a short game from starting position, returning the sequence of (state, move) pairs.
// The game: player 0 takes 1, player 1 takes 2, player 0 takes 1, player 1 takes 3...
// We'll play a fixed sequence that produces known states.
struct GameRecord {
  std::vector<State> states;
  std::vector<Move> moves;
  State final_state;
};

GameRecord play_short_game() {
  GameRecord rec;

  State state;
  Rules::init_state(state);

  // Play moves: take 1, take 2, take 1, take 2, ... until game over
  // With 21 stones and alternating take-1, take-2 = removing 3 per round
  // 21 / 3 = 7 rounds => 14 moves total, but we'll play just a few moves
  Move moves_seq[] = {Move(0), Move(1), Move(2), Move(0), Move(1)};
  int n_moves = 5;  // take 1+2+3+1+2 = 9 stones, leaving 12

  for (int i = 0; i < n_moves; ++i) {
    rec.states.push_back(state);
    rec.moves.push_back(moves_seq[i]);
    Rules::apply(state, moves_seq[i]);
  }

  rec.final_state = state;
  return rec;
}

// Build a GameWriteLog and serialize it.
// Returns metadata + data_buf, plus the total sample count.
struct SerializedGame {
  GameLogMetadata metadata;
  std::vector<char> data_buf;
  int num_samples;
  GameResultTensor outcome;
  // Store the training infos for verification after round-trip
  std::vector<TrainingInfo> training_infos;
};

SerializedGame build_and_serialize_game(const GameRecord& game, float policy_base,
                                        float av_base) {
  SerializedGame result;

  GameWriteLog write_log(/*id=*/1, /*start_timestamp=*/12345);

  int n_moves = game.states.size();
  for (int i = 0; i < n_moves; ++i) {
    // Use different fill values per move so rows are distinguishable after shuffle
    float policy_fill = policy_base + 0.01f * i;
    float av_fill = av_base + 0.01f * i;

    // Mark all as use_for_training except potentially the last one
    bool use = true;
    TrainingInfo info = make_training_info(game.states[i], game.moves[i], policy_fill, av_fill,
                                           use);

    // For the opponent-policy target to work, the *next* record's policy must also be valid.
    // This is handled automatically since we set policy_target_valid=true for all entries.

    write_log.add(info);
    result.training_infos.push_back(info);
  }

  // Set outcome: player 0 wins
  result.outcome.setValues({1.0f, 0.0f});
  write_log.add_terminal(game.final_state, result.outcome);

  GameLogSerializer serializer;
  result.metadata = serializer.serialize(&write_log, result.data_buf, /*client_id=*/0);
  result.num_samples = write_log.sample_count();

  return result;
}

// Compute expected row_size for a given set of target indices
int compute_row_size(const std::vector<int>& target_indices) {
  // Input size
  int row_size = InputTensor::Dimensions::total_size;  // 21

  for (int idx : target_indices) {
    // Each target contributes its tensor size + 1 (mask)
    if (idx == 0) row_size += PolicyTensor::Dimensions::total_size + 1;       // 3 + 1
    else if (idx == 1) row_size += GameResultTensor::Dimensions::total_size + 1;  // 2 + 1
    else if (idx == 2) row_size += ActionValueTensor::Dimensions::total_size + 1; // 6 + 1
    else if (idx == 3) row_size += PolicyTensor::Dimensions::total_size + 1;      // 3 + 1 (opp)
  }
  return row_size;
}

// ============================================================================
// Test 1: Serialize -> Deserialize via GameLogFileReader + GameReadLog
// ============================================================================
TEST(GameLogRoundTrip, SerializeDeserialize) {
  GameRecord game = play_short_game();
  SerializedGame sg = build_and_serialize_game(game, /*policy_base=*/0.5f, /*av_base=*/0.1f);

  // Assemble file buffer
  std::vector<GameLogMetadata> metadata_vec = {sg.metadata};
  int64_t buf_size;
  char* buf = assemble_file_buffer(metadata_vec, sg.data_buf, sg.num_samples, buf_size);

  // Read back
  GameLogFileReader reader(buf);
  ASSERT_EQ(reader.num_games(), 1);
  ASSERT_EQ(reader.num_samples(0), sg.num_samples);

  // Load each row and verify
  std::vector<int> target_indices = {0, 1, 2, 3};  // all 4 targets
  int row_size = compute_row_size(target_indices);

  GameReadLog read_log("test", 0, reader.metadata(0), reader.game_data_buffer(0));
  ASSERT_EQ(read_log.num_sampled_frames(), sg.num_samples);

  for (int row = 0; row < sg.num_samples; ++row) {
    std::vector<float> output(row_size, -999.0f);
    read_log.load(row, /*apply_symmetry=*/false, target_indices, output.data());

    // Verify input tensor: for Nim, encode() sets 1s for the last stones_left positions
    const TrainingInfo& expected_info = sg.training_infos[row];
    InputEncoder encoder;
    encoder.restore(&expected_info.frame, 1);
    InputTensor expected_input = encoder.encode();

    constexpr int kInputSize = InputTensor::Dimensions::total_size;
    for (int i = 0; i < kInputSize; ++i) {
      EXPECT_FLOAT_EQ(output[i], expected_input.data()[i])
        << "Input mismatch at row=" << row << " i=" << i;
    }

    // Verify policy target (kName="policy")
    int offset = kInputSize;
    const PolicyTensor& expected_policy = expected_info.policy_target;
    for (int i = 0; i < PolicyTensor::Dimensions::total_size; ++i) {
      EXPECT_FLOAT_EQ(output[offset + i], expected_policy.data()[i])
        << "Policy mismatch at row=" << row << " i=" << i;
    }
    offset += PolicyTensor::Dimensions::total_size;
    EXPECT_FLOAT_EQ(output[offset], 1.0f) << "Policy mask should be 1 (valid) at row=" << row;
    offset++;

    // Verify value target: game_result left-rotated by active_seat
    GameResultTensor expected_value = sg.outcome;
    TensorEncodings::GameResultEncoding::left_rotate(expected_value, expected_info.active_seat);
    for (int i = 0; i < GameResultTensor::Dimensions::total_size; ++i) {
      EXPECT_FLOAT_EQ(output[offset + i], expected_value.data()[i])
        << "Value mismatch at row=" << row << " i=" << i;
    }
    offset += GameResultTensor::Dimensions::total_size;
    EXPECT_FLOAT_EQ(output[offset], 1.0f) << "Value mask should be 1 at row=" << row;
    offset++;

    // Verify action_value target: left-rotated by active_seat
    ActionValueTensor expected_av = expected_info.action_values_target;
    eigen_util::left_rotate(expected_av, expected_info.active_seat);
    for (int i = 0; i < ActionValueTensor::Dimensions::total_size; ++i) {
      EXPECT_FLOAT_EQ(output[offset + i], expected_av.data()[i])
        << "ActionValue mismatch at row=" << row << " i=" << i;
    }
    offset += ActionValueTensor::Dimensions::total_size;
    EXPECT_FLOAT_EQ(output[offset], 1.0f) << "ActionValue mask should be 1 at row=" << row;
    offset++;

    // OppPolicy target: next_record's policy if available, else mask=0
    // For all rows except the last, the next record's policy should be valid
    if (row + 1 < (int)sg.training_infos.size()) {
      const PolicyTensor& expected_opp = sg.training_infos[row + 1].policy_target;
      for (int i = 0; i < PolicyTensor::Dimensions::total_size; ++i) {
        EXPECT_FLOAT_EQ(output[offset + i], expected_opp.data()[i])
          << "OppPolicy mismatch at row=" << row << " i=" << i;
      }
      offset += PolicyTensor::Dimensions::total_size;
      EXPECT_FLOAT_EQ(output[offset], 1.0f) << "OppPolicy mask should be 1 at row=" << row;
    } else {
      // last training row: next_record is nullptr => opp_policy mask=0
      offset += PolicyTensor::Dimensions::total_size;
      EXPECT_FLOAT_EQ(output[offset], 0.0f) << "OppPolicy mask should be 0 for last row";
    }
    offset++;

    EXPECT_EQ(offset, row_size);
  }

  delete[] buf;
}

// ============================================================================
// Test 2: DataLoader round-trip (parameterized over threading/memory configs)
// ============================================================================
struct DataLoaderConfig {
  int num_workers;
  int num_prefetch;
  int64_t memory_budget;
};

class DataLoaderRoundTrip : public ::testing::TestWithParam<DataLoaderConfig> {};

TEST_P(DataLoaderRoundTrip, LoadMatchesSerializedData) {
  const DataLoaderConfig& cfg = GetParam();

  // Build two games with distinguishable data
  GameRecord game1 = play_short_game();
  GameRecord game2 = play_short_game();  // same game sequence, different fill values

  SerializedGame sg1 = build_and_serialize_game(game1, /*policy_base=*/0.5f, /*av_base=*/0.1f);
  SerializedGame sg2 = build_and_serialize_game(game2, /*policy_base=*/0.8f, /*av_base=*/0.3f);

  int total_rows = sg1.num_samples + sg2.num_samples;

  // Assemble file buffer for gen 1 (game 1 + game 2 in same file)
  std::vector<GameLogMetadata> metadata_vec = {sg1.metadata, sg2.metadata};

  // Combine data buffers; adjust sg2's start_offset
  std::vector<char> combined_data = sg1.data_buf;
  GameLogMetadata md2_adjusted = sg2.metadata;
  md2_adjusted.start_offset += sg1.data_buf.size();
  metadata_vec[1] = md2_adjusted;
  combined_data.insert(combined_data.end(), sg2.data_buf.begin(), sg2.data_buf.end());

  int64_t buf_size;
  char* buf = assemble_file_buffer(metadata_vec, combined_data, total_rows, buf_size);

  // Create DataLoader. data_dir is irrelevant since we inject buffers.
  search::DataLoaderBase::Params params{"unused", cfg.memory_budget, cfg.num_workers,
                                        cfg.num_prefetch};
  search::DataLoader<NimSpec> loader(params);

  loader.test_add_gen_from_buffer(/*gen=*/1, total_rows, buf, buf_size);

  // Load all rows
  std::vector<int> target_indices = {0, 1, 2, 3};
  int row_size = compute_row_size(target_indices);
  std::vector<float> output(total_rows * row_size, -999.0f);
  int gen_range[2] = {0, 0};
  int version_check[2] = {0, 0};

  search::DataLoaderBase::LoadParams load_params;
  load_params.window_start = 0;
  load_params.window_end = total_rows;
  load_params.n_samples = total_rows;
  load_params.apply_symmetry = false;
  load_params.n_targets = kNumTargets;
  load_params.output_array = output.data();
  load_params.target_indices_array = target_indices.data();
  load_params.gen_range = gen_range;
  load_params.version_check = version_check;

  loader.load(load_params);

  EXPECT_EQ(version_check[0], 0);  // 0 means no version mismatch
  EXPECT_EQ(version_check[1], 0);

  // Build reference data: independently read via GameLogFileReader + GameReadLog
  // (using the same buffer) to get expected rows
  char* ref_buf = new char[buf_size];
  std::memcpy(ref_buf, buf, buf_size);

  GameLogFileReader reader(ref_buf);
  ASSERT_EQ(reader.num_games(), 2);

  std::vector<std::vector<float>> expected_rows;
  for (int g = 0; g < reader.num_games(); ++g) {
    GameReadLog read_log("ref", g, reader.metadata(g), reader.game_data_buffer(g));
    for (int r = 0; r < read_log.num_sampled_frames(); ++r) {
      std::vector<float> row(row_size);
      read_log.load(r, false, target_indices, row.data());
      expected_rows.push_back(row);
    }
  }
  delete[] ref_buf;

  ASSERT_EQ((int)expected_rows.size(), total_rows);

  // DataLoader samples randomly (with replacement), so each output row should match some expected
  // row. Verify every actual row is found in the expected set.
  auto rows_match = [&](const std::vector<float>& a, const std::vector<float>& b) {
    for (int i = 0; i < row_size; ++i) {
      if (a[i] != b[i]) return false;
    }
    return true;
  };

  for (int i = 0; i < total_rows; ++i) {
    std::vector<float> actual_row(output.begin() + i * row_size,
                                  output.begin() + (i + 1) * row_size);
    bool found = false;
    for (const auto& expected_row : expected_rows) {
      if (rows_match(actual_row, expected_row)) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Row " << i << " from DataLoader does not match any expected row";
  }
}

INSTANTIATE_TEST_SUITE_P(Configs, DataLoaderRoundTrip,
  ::testing::Values(
    DataLoaderConfig{1, 0, 1 << 30},   // 1 worker, 0 prefetch, large memory
    DataLoaderConfig{2, 0, 1 << 30},   // 2 workers, 0 prefetch, large memory
    DataLoaderConfig{4, 0, 1 << 30},   // 4 workers, 0 prefetch, large memory
    DataLoaderConfig{1, 1, 1 << 30},   // 1 worker, 1 prefetch, large memory
    DataLoaderConfig{2, 2, 1 << 30},   // 2 workers, 2 prefetch, large memory
    DataLoaderConfig{1, 0, 256}        // 1 worker, 0 prefetch, tiny memory budget
  ),
  [](const ::testing::TestParamInfo<DataLoaderConfig>& p) {
    return "w" + std::to_string(p.param.num_workers) +
           "_p" + std::to_string(p.param.num_prefetch) +
           "_m" + std::to_string(p.param.memory_budget);
  }
);
