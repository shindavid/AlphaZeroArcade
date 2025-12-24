#include "util/BoostUtil.hpp"

#include "util/Random.hpp"

#include <format>

namespace boost_util {

int get_random_set_index(std::mt19937& prng, const boost::dynamic_bitset<>& bitset) {
  int count = bitset.count();

  if (count == 0) {
    return -1;  // No bits are set
  }

  int n = util::Random::uniform_sample(prng, 0, count);

  int index = bitset.find_first();
  while (n-- > 0) {
    index = bitset.find_next(index);
  }

  return index;
}

int get_random_set_index(const boost::dynamic_bitset<>& bitset) {
  return get_random_set_index(util::Random::default_prng(), bitset);
}

std::string get_option_value(const std::vector<std::string>& args, const std::string& option_name) {
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

std::string pop_option_value(std::vector<std::string>& args, const std::string& option_name) {
  std::string dashed_option_name = "--" + option_name;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg == dashed_option_name) {
      if (i + 1 >= args.size()) {
        throw std::runtime_error(std::format("Missing value for option '{}'", option_name.c_str()));
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

// The code below was adapted from:
// https://www.boost.org/doc/libs/1_76_0/libs/json/doc/html/json/examples.html
void pretty_print(std::ostream& os, boost::json::value const& jv, std::string* indent) {
  std::string indent_;
  if (!indent) indent = &indent_;
  switch (jv.kind()) {
    case boost::json::kind::object: {
      os << "{\n";
      auto const& obj = jv.get_object();

      if (!obj.empty()) {
        // Collect iterators, sort by key
        std::vector<boost::json::object::const_iterator> its;
        its.reserve(obj.size());
        for (auto it = obj.begin(); it != obj.end(); ++it) its.push_back(it);

        std::sort(its.begin(), its.end(), [](auto a, auto b) { return a->key() < b->key(); });

        // Print in sorted order
        for (std::size_t i = 0; i < its.size(); ++i) {
          auto it = its[i];
          os << *indent << boost::json::serialize(it->key()) << " : ";
          pretty_print(os, it->value(), indent);
          if (i + 1 != its.size()) os << ",\n";
        }
      }

      os << "\n";
      os << *indent << "}";
      break;
    }

    case boost::json::kind::array: {
      auto const& arr = jv.get_array();
      auto it = arr.begin();
      bool is_simple_array =
        (it->kind() != boost::json::kind::object) && (it->kind() != boost::json::kind::array);

      // print without newlines if the array contains only simple elements
      if (is_simple_array) {
        os << "[";

        while (true) {
          pretty_print(os, *it, indent);
          if (++it == arr.end()) break;
          os << ", ";
        }
        os << "]";
      } else {
        os << "[\n";
        indent->append(2, ' ');

        while (true) {
          os << *indent;
          pretty_print(os, *it, indent);
          if (++it == arr.end()) break;
          os << ",\n";
        }
        os << "\n";
        indent->resize(indent->size() - 2);
        os << *indent << "]";
      }
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

    case boost::json::kind::double_: {
      auto x = jv.get_double();
      if (x == 0) {  // IEEE 754 standard is weird, 0 can be printed as -0
        os << "0";
      } else {
        os << x;
      }
      break;
    }

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
}

}  // namespace boost_util
