#pragma once

#include <cstdint>
#include <string>

namespace core {

const int kMaxNameLength = 32;  // excluding null terminator

/*
 * All Game objects must serialize to representations of this size or smaller.
 *
 * If we introduce a game that cannot respect this limit, we can either increase this limit, or we
 * can templatize the classes in Packet.hpp by Game, and have them use a different
 * serialization limit for each game.
 *
 * Note that there is no real performance overhead associated with setting this value too high. In
 * various spots we declare char buffers of this size on the stack, but we only read/write as many
 * bytes as we need to from them.
 */
const int kSerializationLimit = 1024;

// See KataGo paper for description of search modes.
enum SearchMode : int8_t { kFast, kFull, kRawPolicy, kNumSearchModes };

constexpr int kNumRowsToDisplayVerbose = 10;

enum SearchParadigm : int8_t { kParadigmMcts, kParadigmBmcts, kUnknownParadigm };

inline SearchParadigm parse_search_paradigm(const char* s) {
  std::string ss(s);
  if (ss == "mcts") {
    return kParadigmMcts;
  } else if (ss == "bmcts") {
    return kParadigmBmcts;
  } else {
    return kUnknownParadigm;
  }
}

}  // namespace core
