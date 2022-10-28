#include <util/StringUtil.hpp>

#include <boost/algorithm/string.hpp>

inline std::vector<std::string> split(const std::string& s) {
  namespace ba = boost::algorithm;

  std::string delims = " \n\t\r\v\f";  // matches python isspace()
  std::vector<std::string> tokens;
  ba::split(tokens, s, boost::is_any_of(delims));
}
