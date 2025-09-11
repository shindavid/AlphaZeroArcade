#pragma once

#include <boost/filesystem.hpp>

#include <cstdint>
#include <string>

namespace nnet {

/*
 * Controls behavior of an NNEvaluationService.
 */
struct NNEvaluationServiceParams {
  auto make_options_description();
  bool operator==(const NNEvaluationServiceParams& other) const = default;

  std::string model_filename;
  bool no_model = false;
  std::string cuda_device = "cuda:0";
  int num_pipelines = 2;
  size_t cache_size = 1048576;

  int batch_size = 256;
  uint64_t engine_build_workspace_size_in_bytes = 1ULL << 32;  // 4GB
  std::string engine_build_precision = "FP16";

  bool apply_random_symmetries = true;

  /*
   * When we use a neural network to evaluate a position, we first apply a random symmetry to the
   * position.
   *
   * If incorporate_sym_into_cache_key is true, then this sym is part of the cache key.
   *
   * The benefit of incorporating sym into the cache key is that when multiple games are played,
   * those games are independent. The downside is that we get less cache hits, hurting game
   * throughput.
   *
   * @dshin performed ad-hoc tests in 2024 that showed that setting this to false in evaluation
   * games contributed unacceptably large noise in rating evaluations. Based on that, we set it to
   * true for kCompetition mode.
   *
   * In August 2025, @lichensongs performed further ad-hoc tests in the games of hex and Othello.
   * These tests showed that in hex, incorporating sym into the cache key significantly decreased
   * variance in the the skill-curve. Furthermore, though the cache hit rate was indeed lower, the
   * overall skill progression with respect to self-play-time actually improved, for reasons we
   * don't yet completely understand. In Othello, we observed a slight reduction in skill-curve
   * variance, and no visible change in skill progression with respect to self-play-time.
   *
   * Skill-curve variance reduction is a worthy goal, as it allows us to perform A/B tests more
   * effectively.
   *
   * Based on these tests, we decided to set this bool to true in both modes.
   */
  bool incorporate_sym_into_cache_key = true;
};

}  // namespace nnet

#include "inline/nnet/NNEvaluationServiceParams.inl"
