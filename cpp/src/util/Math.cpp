#include <util/Math.hpp>

#include <vector>

#include <tinyexpr.h>

#include <util/Exception.hpp>
#include <util/StringUtil.hpp>

namespace math {

ExponentialDecay::ExponentialDecay(float value) : ExponentialDecay(value, value, 1.0) {}

ExponentialDecay::ExponentialDecay(float start_value, float end_value, float half_life)
    : start_value_(start_value),
      end_value_(end_value),
      half_life_(half_life),
      decay_factor_(half_life_to_decay(half_life)),
      cur_value_(start_value) {}

ExponentialDecay ExponentialDecay::parse(const std::string& repr,
                                                const var_bindings_map_t& bindings) {
  try {
    size_t arrow = repr.find("->");
    if (arrow == std::string::npos) {
      return ExponentialDecay(parse_expression(repr.c_str(), bindings));
    }

    size_t colon = repr.find(':');
    if (colon == std::string::npos) {
      throw std::exception();
    }
    double start_value = parse_expression(repr.substr(0, arrow).c_str(), bindings);
    double end_value =
        parse_expression(repr.substr(arrow + 2, colon - arrow - 2).c_str(), bindings);
    double half_life = parse_expression(repr.substr(colon + 1).c_str(), bindings);

    return ExponentialDecay(start_value, end_value, half_life);
  } catch (...) {
    throw util::Exception("Parse failure - ExponentialDecay::(\"%s\")", repr.c_str());
  }
}

double parse_expression(const char* expr, const var_bindings_map_t& bindings) {
  using variable_vec_t = std::vector<te_variable>;

  variable_vec_t vars;
  for (const auto& it : bindings) {
    te_variable var{it.first.c_str(), &it.second};
    vars.push_back(var);
  }

  int err;
  te_expr* compiled_expr = te_compile(expr, vars.data(), (int)vars.size(), &err);
  if (compiled_expr) {
    double result = te_eval(compiled_expr);
    te_free(compiled_expr);
    return result;
  } else {
    fprintf(stderr, "%s\n", expr);
    fprintf(stderr, "\t%*s^\nError near here", err - 1, "");
    throw util::Exception("%s(\"%s\") failure", __func__, expr);
  }
}

}  // namespace math
