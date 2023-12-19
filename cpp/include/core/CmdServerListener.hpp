#pragma once

#include <core/PerfStats.hpp>
#include <util/Exception.hpp>

#include <boost/json.hpp>

namespace core {

enum class CmdServerMsgType { kPause, kReloadWeights, kMetricsRequest, kFlushGames };

class CmdServerClient;

/*
 * A connection to a cmd-server can be initiated via core::CmdServerClient::init(). Once that
 * connection is established, any number of CmdServerListeners can register with the singleton
 * core::CmdServerClient.
 */
template <CmdServerMsgType type>
class CmdServerListener {};

template <>
class CmdServerListener<CmdServerMsgType::kPause> {
 public:
  friend class CmdServerClient;

  virtual ~CmdServerListener() = default;
  virtual void pause() = 0;
  virtual void unpause() = 0;

 private:
  bool ready_for_pause_ack_ = false;
};

template <>
class CmdServerListener<CmdServerMsgType::kReloadWeights> {
 public:
  virtual ~CmdServerListener() = default;
  virtual void reload_weights(const std::string& model_filename) = 0;
};

template <>
class CmdServerListener<CmdServerMsgType::kMetricsRequest> {
 public:
  virtual ~CmdServerListener() = default;
  virtual perf_stats_t get_perf_stats() = 0;
};

template <>
class CmdServerListener<CmdServerMsgType::kFlushGames> {
 public:
  friend class CmdServerClient;

  virtual ~CmdServerListener() = default;
  virtual void flush_games(int next_generation) = 0;

 protected:
  bool ready_for_flush_games_ack_ = false;
};

}  // namespace core
