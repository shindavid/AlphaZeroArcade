#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <initializer_list>
#include <vector>

using int_vec_t = std::vector<int>;

int_vec_t concatenate() { return {}; }

//template<typename... Ts> int_vec_t concatenate(Ts&&... ts);

//template<typename... Ts, typename T>
//int_vec_t concatenate(Ts&&... ts, const std::initializer_list<T>& t) {
//  int_vec_t vec = concatenate(std::forward<Ts>(ts)...);
//  vec.insert(vec.end(), t);
//  return vec;
//}

template<typename T, typename... Ts>
int_vec_t concatenate(T t, Ts&&... ts) {
  int_vec_t vec1 = {t};
  int_vec_t vec2 = concatenate(std::forward<Ts>(ts)...);
  vec1.insert(vec1.end(), vec2.begin(), vec2.end());
  return vec1;
}

int main(int ac, char* av[]) {
  auto v1 = concatenate(1, 2, 3, 4);
  for (auto t : v1) {
    std::cout << t << std::endl;
  }
  return 0;
}
