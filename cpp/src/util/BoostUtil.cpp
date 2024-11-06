#include <util/BoostUtil.hpp>

#include <boost/json.hpp>

namespace boost_util {

std::string get_option_value(const std::vector<std::string>& args,
                                    const std::string& option_name) {
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

std::string pop_option_value(std::vector<std::string>& args,
                                    const std::string& option_name) {
  std::string dashed_option_name = "--" + option_name;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg == dashed_option_name) {
      if (i + 1 >= args.size()) {
        throw std::runtime_error(
            util::create_string("Missing value for option '%s'", option_name.c_str()));
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

// The code below was taken from:
// https://www.boost.org/doc/libs/1_76_0/libs/json/doc/html/json/examples.html
void pretty_print(std::ostream& os, boost::json::value const& jv, std::string* indent) {
  std::string indent_;
  if (!indent) indent = &indent_;
  switch (jv.kind()) {
    case boost::json::kind::object: {
      os << "{\n";
      auto const& obj = jv.get_object();
      if (!obj.empty()) {
        auto it = obj.begin();
        for (;;) {
          os << *indent << boost::json::serialize(it->key()) << " : ";
          pretty_print(os, it->value(), indent);
          if (++it == obj.end()) break;
          os << ",\n";
        }
      }
      os << "\n";
      os << *indent << "}";
      break;
    }

    case boost::json::kind::array: {
      os << "[\n";
      indent->append(4, ' ');
      auto const& arr = jv.get_array();
      if (!arr.empty()) {
        auto it = arr.begin();
        for (;;) {
          os << *indent;
          pretty_print(os, *it, indent);
          if (++it == arr.end()) break;
          os << ",\n";
        }
      }
      os << "\n";
      indent->resize(indent->size() - 4);
      os << *indent << "]";
      break;
    }

    case boost::json::kind::string: {
      os << boost::json::serialize(jv.get_string());
      break;
    }

    case boost::json::kind::uint64:
      os << jv.get_uint64();
      break;

    case boost::json::kind::int64:
      os << jv.get_int64();
      break;

    case boost::json::kind::double_:
      os << jv.get_double();
      break;

    case boost::json::kind::bool_:
      if (jv.get_bool())
        os << "true";
      else
        os << "false";
      break;

    case boost::json::kind::null:
      os << "null";
      break;
  }

  if (indent->empty()) os << "\n";
}

namespace program_options {

bool Settings::help_full = false;

}  // namespace program_options

}  // namespace boost_util
