#pragma once

#include <concepts>
#include <string>
#include <string_view>

namespace core {

namespace concepts {

/*
 * All Game::Move types M must satisfy core::concepts::Move<M, State>.
 *
 * Requirements:
 * - Default- and copy-constructible
 * - POD (standard-layout + trivially copyable)
 * - to_str() -> std::string
 * - static from_str(const State&, std::string_view) -> M
 */
template <class M, class State>
concept Move = std::is_default_constructible_v<M> && std::is_copy_constructible_v<M> &&
               std::is_standard_layout_v<M> && std::is_trivially_copyable_v<M> &&
               requires(const M& a, const M& b, const State& s, std::string_view sv) {
                 { a.to_str() } -> std::same_as<std::string>;
                 { M::from_str(s, sv) } -> std::same_as<M>;
               };

}  // namespace concepts

}  // namespace core
