#pragma once

#include <core/CmdServerListener.hpp>
#include <core/PerfStats.hpp>
#include <util/CppUtil.hpp>
#include <util/SocketUtil.hpp>

#include <boost/json.hpp>

#include <string>
#include <thread>
#include <vector>

namespace core {

/*
 * CmdServerClient is used to communicate with an external cmd-server. It is to be used as a
 * singleton, and can further forward messages to any number of listeners.
 *
 * For now, all messages will be in json format. We can revisit this in the future.
 *
 * Usage:
 *
 * core::CmdServerClient::init(host, port);
 *
 * core::CmdServerClient* client = core::CmdServerClient::get();
 * client->send("abc", 3);
 *
 * To listen to messages from the cmd-server, implement the CmdServerListener interface and
 * subscribe to the client via client->add_listener(listener).
 */
class CmdServerClient {
 public:
  static void init(const std::string& host, io::port_t port);
  static bool initialized() { return instance_;  }
  static CmdServerClient* get() { return instance_;  }
  int client_id() const { return client_id_; }

  template <typename T>
  void add_listener(T* listener);

  void send(const boost::json::value& msg) { socket_->json_write(msg); }
  bool ready_for_pause_ack();
  void pause_ack();
  bool ready_for_flush_games_ack();
  void flush_games_ack();
  perf_stats_t get_perf_stats() const;

  int64_t get_last_metrics_ts() const { return last_metrics_ts_; }
  bool ready_for_metrics(int64_t ts) const { return ts > last_metrics_ts_ + util::s_to_ns(1); }
  void set_last_metrics_ts(int64_t ts) { last_metrics_ts_ = ts; }

 private:
  using PauseListener = CmdServerListener<CmdServerMsgType::kPause>;
  using ReloadWeightsListener = CmdServerListener<CmdServerMsgType::kReloadWeights>;
  using MetricsRequestListener = CmdServerListener<CmdServerMsgType::kMetricsRequest>;
  using FlushGamesListener = CmdServerListener<CmdServerMsgType::kFlushGames>;

  CmdServerClient(const std::string& host, io::port_t port);
  ~CmdServerClient();
  void send_handshake();
  void recv_handshake();
  void handle_pause();
  void handle_reload_weights(const std::string& model_filename);
  void handle_metrics_request();
  void handle_flush_games(int next_generation);
  void loop();

  static CmdServerClient* instance_;

  const int64_t proc_start_ts_;
  int64_t last_metrics_ts_ = 0;
  io::Socket* socket_;
  std::thread* thread_;
  std::vector<PauseListener*> pause_listeners_;
  std::vector<ReloadWeightsListener*> reload_weights_listeners_;
  std::vector<MetricsRequestListener*> metrics_request_listeners_;
  std::vector<FlushGamesListener*> flush_games_listeners_;
  int client_id_ = -1;  // assigned by cmd-server
};

}  // namespace core

#include <inline/core/CmdServerClient.inl>
