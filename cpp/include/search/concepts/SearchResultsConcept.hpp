#pragma once

#include <boost/json.hpp>

#include <concepts>

namespace search {
namespace concepts {

template <class S>
concept SearchResults = requires(const S& results) {
  requires std::default_initializable<S>;
  { results.to_json() } -> std::same_as<boost::json::object>;
};

}  // namespace concepts
}  // namespace search
