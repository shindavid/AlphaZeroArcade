#include <mcts/NNEvaluationServiceParams.hpp>

#include <boost/filesystem.hpp>
#include <util/BoostUtil.hpp>
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
          "model filename. If not specified, use an implicit uniform model if --no-model is set. "
          "If neither --model-filename nor --no-model are set, then will receive model from "
          "loop-controller")
      .template add_flag<"no-model", "use-model">(&no_model, "No model (uniform evals)",
                                                  "Use model for evals")
      .template add_option<"cuda-device">(
          po::value<std::string>(&cuda_device)->default_value(cuda_device),
          "cuda device to use for nn evals. Usually you need to specify this again outside the "
          "--player string to register the device to the loop controller")
      .template add_option<"batch-size-limit", 'b'>(
          po::value<int>(&batch_size_limit)->default_value(batch_size_limit), "batch size limit")
      .template add_hidden_option<"cache-size">(
          po::value<size_t>(&cache_size)->default_value(cache_size), "nn eval thread cache size")
#ifdef PROFILE_MCTS
      .template add_option<"profiling-dir">(
          po::value<std::string>(&profiling_dir_str)->default_value("/workspace/repo/profiling"),
          "directory in which to dump mcts profiling stats")
#endif  // PROFILE_MCTS
      ;
}

}  // namespace mcts
