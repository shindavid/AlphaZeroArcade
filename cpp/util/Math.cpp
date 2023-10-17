#include <util/Math.hpp>

#include <vector>

#include <tinyexpr.h>

#include <util/Exception.hpp>

namespace math {

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
    printf("%s\n", expr);
    printf("\t%*s^\nError near here", err - 1, "");
    throw util::Exception("%s(\"%s\") failure", __func__, expr);
  }
}

}  // namespace math
