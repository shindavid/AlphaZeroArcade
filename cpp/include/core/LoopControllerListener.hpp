#pragma once

#include <core/PerfStats.hpp>
#include <util/Exception.hpp>

#include <sstream>

#include <boost/json.hpp>

namespace core {

enum class LoopControllerInteractionType { kPause, kReloadWeights, kMetricsRequest };

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

 private:
  bool pause_notified_ = false;  // used by LoopControllerClient
};

template <>
class LoopControllerListener<LoopControllerInteractionType::kReloadWeights> {
 public:
  virtual ~LoopControllerListener() = default;
  virtual void reload_weights(std::stringstream&, const std::string& cuda_device) = 0;
};

template <>
class LoopControllerListener<LoopControllerInteractionType::kMetricsRequest> {
 public:
  virtual ~LoopControllerListener() = default;
  virtual perf_stats_t get_perf_stats() = 0;
};

}  // namespace core
