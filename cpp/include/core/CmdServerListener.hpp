#pragma once

#include <core/PerfStats.hpp>
#include <util/Exception.hpp>

#include <boost/json.hpp>

namespace core {

enum class CmdServerInteractionType {
  kPause,
  kReloadWeights,
  kMetricsRequest,
  kUpdateGeneration
};

class CmdServerClient;

/*
 * A connection to a cmd-server can be initiated via core::CmdServerClient::init(). Once that
 * connection is established, any number of CmdServerListeners can register with the singleton
 * core::CmdServerClient.
 *
 * When the CmdServerClient receives a message from the cmd-server, it engages in various
 * interactions with the registered listeners. The interaction types are defined in
 * the CmdServerInteractionType enum. These interaction types are not in general the same as the
 * msg types sent by the cmd-server.
 */
template <CmdServerInteractionType type>
class CmdServerListener {};

template <>
class CmdServerListener<CmdServerInteractionType::kPause> {
 public:
  friend class CmdServerClient;

  virtual ~CmdServerListener() = default;
  virtual void pause() = 0;
  virtual void unpause() = 0;

 private:
  bool pause_notified_ = false;  // used by CmdServerClient
};

template <>
class CmdServerListener<CmdServerInteractionType::kReloadWeights> {
 public:
  virtual ~CmdServerListener() = default;
  virtual void reload_weights(const std::string& model_filename) = 0;
};

template <>
class CmdServerListener<CmdServerInteractionType::kMetricsRequest> {
 public:
  virtual ~CmdServerListener() = default;
  virtual perf_stats_t get_perf_stats() = 0;
};

template <>
class CmdServerListener<CmdServerInteractionType::kUpdateGeneration> {
 public:
  virtual ~CmdServerListener() = default;
  virtual void update_generation(int generation) = 0;
};

}  // namespace core
