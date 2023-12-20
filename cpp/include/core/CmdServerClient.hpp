#pragma once

#include <core/CmdServerListener.hpp>
#include <core/PerfStats.hpp>
#include <util/CppUtil.hpp>
#include <util/SocketUtil.hpp>

#include <boost/json.hpp>

#include <condition_variable>
#include <mutex>
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
 * core::CmdServerClient::Params params;
 * // set params
 * core::CmdServerClient::init(params);
 *
 * core::CmdServerClient* client = core::CmdServerClient::get();
 * client->send("abc", 3);
 *
 * To listen to messages from the cmd-server, implement the CmdServerListener interface and
 * subscribe to the client via client->add_listener(listener).
 */
class CmdServerClient {
 public:
  struct Params {
    auto make_options_description();
    bool operator==(const Params& other) const = default;

    std::string cmd_server_ip_addr;
    io::port_t cmd_server_port = 0;
    bool shared_gpu = false;
  };

  using PauseListener = CmdServerListener<CmdServerInteractionType::kPause>;
  using ReloadWeightsListener = CmdServerListener<CmdServerInteractionType::kReloadWeights>;
  using MetricsRequestListener = CmdServerListener<CmdServerInteractionType::kMetricsRequest>;
  using UpdateGenerationListener = CmdServerListener<CmdServerInteractionType::kUpdateGeneration>;

  static void init(const Params&);
  static bool initialized() { return instance_;  }
  static CmdServerClient* get() { return instance_;  }
  int client_id() const { return client_id_; }

  template <typename T>
  void add_listener(T* listener);

  void send(const boost::json::value& msg) { socket_->json_write(msg); }

  void notify_pause_received(PauseListener* listener);
  perf_stats_t get_perf_stats() const;

  int64_t get_last_metrics_ts() const { return last_metrics_ts_; }
  bool ready_for_metrics(int64_t ts) const { return ts > last_metrics_ts_ + util::s_to_ns(1); }
  void set_last_metrics_ts(int64_t ts) { last_metrics_ts_ = ts; }

 private:
  CmdServerClient(const Params&);
  ~CmdServerClient();

  void send_handshake();
  void recv_handshake();
  void pause();
  void send_pause_ack();
  void update_generation(int generation);
  void unpause();
  void reload_weights(const std::string& model_filename);
  void send_reload_weights_ack(const perf_stats_t& stats);
  void loop();
  bool all_pause_notifications_received() const;

  static CmdServerClient* instance_;

  const int64_t proc_start_ts_;
  const bool shared_gpu_;
  int64_t last_metrics_ts_ = 0;
  io::Socket* socket_;
  std::thread* thread_;
  std::vector<PauseListener*> pause_listeners_;
  std::vector<ReloadWeightsListener*> reload_weights_listeners_;
  std::vector<MetricsRequestListener*> metrics_request_listeners_;
  std::vector<UpdateGenerationListener*> update_generation_listeners_;
  int client_id_ = -1;  // assigned by cmd-server

  std::condition_variable pause_cv_;
  mutable std::mutex pause_mutex_;
  bool pause_complete_ = false;
};

}  // namespace core

#include <inline/core/CmdServerClient.inl>
