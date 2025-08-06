#include "util/GTestUtil.hpp"

#include "util/LoggingUtil.hpp"
#include "util/Rendering.hpp"

#include <boost/program_options.hpp>

#include <iostream>

// This is a bit tricky.
//
// In order to display help options for gtest, we need to call testing::InitGoogleTest() with
// "--help" in argv. However, doing so causes the program to exit immediately after displaying the
// help, which doesn't give us a chance to show help for our own options.
//
// On the other hand, if we parse our own options first via the normal BoostUtil machinery, we will
// get an unknown-option error if we pass any gtest options.
//
// Therefore, we do a specialized scan for "--help" first. If we find it, we print our help, and
// then call the program-exiting testing::InitGoogleTest() call.
int launch_gtest(int argc, char** argv) {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;
  util::Logging::Params log_params;

  po2::options_description raw_desc("Options");
  auto desc = raw_desc.template add_option<"help", 'h'>("help (most used options)")
                .template add_option<"help-full">("help (all options)")
                .add(log_params.make_options_description());

  bool help_full = false;
  bool help = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--help-full") {
      help_full = true;
    } else if (std::string(argv[i]) == "--help") {
      help = true;
    }
  }
  if (help || help_full) {
    po2::Settings::help_full = help_full;
    std::cout << desc << std::endl;
  }
  if (help_full && !help) {
    // If we had --help-full, rewrite it to --help and then pass it to gtest.
    argc = 2;
    argv[1] = const_cast<char*>("--help");
  }

  testing::InitGoogleTest(&argc, argv);

  po::variables_map vm = po2::parse_args(desc, argc, argv);
  util::Logging::init(log_params);
  util::Rendering::set(util::Rendering::kText);
  return RUN_ALL_TESTS();
}
