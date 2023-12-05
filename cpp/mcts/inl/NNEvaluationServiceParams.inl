#include <mcts/NNEvaluationServiceParams.hpp>

#include <boost/filesystem.hpp>

#include <util/BoostUtil.hpp>
#include <util/Config.hpp>
#include <util/Exception.hpp>
#include <util/RepoUtil.hpp>

namespace mcts {

inline auto NNEvaluationServiceParams::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("Neural net evaluation service options");

  return desc
      .template add_option<"model-filename", 'm'>(
          po::value<std::string>(&model_filename),
          "model filename. If not specified, a uniform model is implicitly used")
      .template add_option<"cuda-device">(
          po::value<std::string>(&cuda_device)->default_value(cuda_device), "cuda device")
      .template add_option<"batch-size-limit", 'b'>(
          po::value<int>(&batch_size_limit)->default_value(batch_size_limit), "batch size limit")
      .template add_option<"nn-eval-timeout-ns">(
          po::value<int64_t>(&nn_eval_timeout_ns)->default_value(nn_eval_timeout_ns),
          "nn eval thread timeout in ns")
      .template add_option<"cache-size">(po::value<size_t>(&cache_size)->default_value(cache_size),
                                         "nn eval thread cache size")
      .template add_flag<"refresh-weights", "do-not-refresh-weights">(
          &refresh_weights, "periodically refresh weights from disk",
          "do not periodically refresh weights from disk");
}

}  // namespace mcts
