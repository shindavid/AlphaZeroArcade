#include <util/BoostUtil.hpp>

namespace boost_util {

inline std::string get_option_value(const std::vector<std::string>& args, const std::string& option_name) {
  std::string dashed_option_name = "--" + option_name;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg == dashed_option_name) {
      if (i + 1 < args.size()) {
        return args[i + 1];
      }
    } else {
      size_t eq_pos = arg.find('=');
      if (eq_pos != std::string::npos) {
        std::string name = arg.substr(0, eq_pos);
        if (name == dashed_option_name) {
          return arg.substr(eq_pos + 1);
        }
      }
    }
  }
  return "";
}

inline std::string pop_option_value(std::vector<std::string>& args, const std::string& option_name) {
  std::string dashed_option_name = "--" + option_name;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg == dashed_option_name) {
      if (i + 1 >= args.size()) {
        throw std::runtime_error(util::create_string("Missing value for option '%s'", option_name.c_str()));
      }
      std::string value = args[i + 1];
      args.erase(args.begin() + i, args.begin() + i + 2);
      return value;
    } else {
      size_t eq_pos = arg.find('=');
      if (eq_pos != std::string::npos) {
        std::string name = arg.substr(0, eq_pos);
        if (name == dashed_option_name) {
          std::string value = arg.substr(eq_pos + 1);
          args.erase(args.begin() + i);
          return value;
        }
      }
    }
  }
  return "";
}

}  // namespace boost_util
