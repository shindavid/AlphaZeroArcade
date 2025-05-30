#pragma once

#include <core/PerfStats.hpp>
#include <util/Exception.hpp>

#include <boost/json.hpp>

#include <vector>

namespace core {

enum class LoopControllerInteractionType {
  kPause,
  kReloadWeights,
  kDataRequest,
  kWorkerReady
};

class LoopControllerClient;

/*
 * A connection to a loop-controller can be initiated via core::LoopControllerClient::init(). Once
 * that connection is established, any number of LoopControllerListeners can register with the
 * singleton core::LoopControllerClient.
 *
 * When the LoopControllerClient receives a message from the loop-controller, it engages in various
 * interactions with the registered listeners. The interaction types are defined in
 * the LoopControllerInteractionType enum. These interaction types are not in general the same as
 * the msg types sent by the loop-controller.
 */
template <LoopControllerInteractionType type>
class LoopControllerListener {};

template <>
class LoopControllerListener<LoopControllerInteractionType::kPause> {
 public:
  friend class LoopControllerClient;

  virtual ~LoopControllerListener() = default;
  virtual void pause() = 0;
  virtual void unpause() = 0;
};

template <>
class LoopControllerListener<LoopControllerInteractionType::kReloadWeights> {
 public:
  virtual ~LoopControllerListener() = default;
  virtual void reload_weights(const std::vector<char>& buf) = 0;
};

template <>
class LoopControllerListener<LoopControllerInteractionType::kDataRequest> {
 public:
  virtual ~LoopControllerListener() = default;

  // Handle a request for n_rows row of self-play data. This data should already have been produced;
  // the loop-controller knows this because it has received heartbeat messages advertising this
  // count.
  virtual void handle_data_request(int n_rows) = 0;

  // The loop-controller is telling us that for the next handle_data_request() call, it will not
  // pass a value of n_rows that exceeds this limit. The listener can use this information to
  // avoid doing unnecessary work.
  virtual void handle_data_pre_request(int n_rows_limit) = 0;
};

template <>
class LoopControllerListener<LoopControllerInteractionType::kWorkerReady> {};

}  // namespace core
