#pragma once

#include <core/PerfStats.hpp>
#include <util/Exception.hpp>

#include <boost/json.hpp>

namespace core {

enum class TrainingServerInteractionType {
  kPause,
  kReloadWeights,
  kMetricsRequest
};

class TrainingServerClient;

/*
 * A connection to a cmd-server can be initiated via core::TrainingServerClient::init(). Once that
 * connection is established, any number of TrainingServerListeners can register with the singleton
 * core::TrainingServerClient.
 *
 * When the TrainingServerClient receives a message from the cmd-server, it engages in various
 * interactions with the registered listeners. The interaction types are defined in
 * the TrainingServerInteractionType enum. These interaction types are not in general the same as the
 * msg types sent by the cmd-server.
 */
template <TrainingServerInteractionType type>
class TrainingServerListener {};

template <>
class TrainingServerListener<TrainingServerInteractionType::kPause> {
 public:
  friend class TrainingServerClient;

  virtual ~TrainingServerListener() = default;
  virtual void pause() = 0;
  virtual void unpause() = 0;

 private:
  bool pause_notified_ = false;  // used by TrainingServerClient
};

template <>
class TrainingServerListener<TrainingServerInteractionType::kReloadWeights> {
 public:
  virtual ~TrainingServerListener() = default;
  virtual void reload_weights(const std::string& model_filename) = 0;
};

template <>
class TrainingServerListener<TrainingServerInteractionType::kMetricsRequest> {
 public:
  virtual ~TrainingServerListener() = default;
  virtual perf_stats_t get_perf_stats() = 0;
};

}  // namespace core
