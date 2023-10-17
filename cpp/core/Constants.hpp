#pragma once

namespace core {

const int kMaxNameLength = 32;  // excluding null terminator

/*
 * All serialize_*() methods of GameState classes must limit their serializations to this size.
 *
 * If we introduce a game that cannot respect this limit, we can either increase this limit, or we
 * can templatize the classes in Packet.hpp by GameState, and have them use a different
 * serialization limit for each game.
 *
 * Note that there is no real performance overhead associated with setting this value too high. In
 * various spots we declare char buffers of this size on the stack, but we only read/write as many
 * bytes as we need to from them.
 */
const int kSerializationLimit = 1024;

}  // namespace core
