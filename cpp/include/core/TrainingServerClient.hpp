#pragma once

#include <core/TrainingServerListener.hpp>
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
 * TrainingServerClient is used to communicate with an external training server. It is to be used as
 * a singleton, and can further forward messages to any number of listeners.
 *
 * For now, all messages will be in json format. We can revisit this in the future.
 *
 * Usage:
 *
 * core::TrainingServerClient::Params params;
 * // set params
 * core::TrainingServerClient::init(params);
 *
 * core::TrainingServerClient* client = core::TrainingServerClient::get();
 * client->send("abc", 3);
 *
 * To listen to messages from the training-server, implement the TrainingServerListener interface
 * and subscribe to the client via client->add_listener(listener).
 */
class TrainingServerClient {
 public:
  struct Params {
    auto make_options_description();
    bool operator==(const Params& other) const = default;

    std::string training_server_hostname = "localhost";
    io::port_t training_server_port = 0;
    int starting_generation = 0;
    std::string cuda_device = "cuda:0";
  };

  using PauseListener = TrainingServerListener<TrainingServerInteractionType::kPause>;
  using ReloadWeightsListener = TrainingServerListener<TrainingServerInteractionType::kReloadWeights>;
  using MetricsRequestListener = TrainingServerListener<TrainingServerInteractionType::kMetricsRequest>;

  static void init(const Params&);
  static bool initialized() { return instance_;  }
  static TrainingServerClient* get() { return instance_;  }
  int client_id() const { return client_id_; }
  int cur_generation() const { return cur_generation_; }

  template <typename T>
  void add_listener(T* listener);

  void send(const boost::json::value& msg) { socket_->json_write(msg); }

  void notify_pause_received(PauseListener* listener);
  perf_stats_t get_perf_stats() const;

  int64_t get_last_games_flush_ts() const { return last_games_flush_ts_; }
  bool ready_for_games_flush(int64_t ts) const {
    return ts > last_games_flush_ts_ + util::s_to_ns(1);
  }
  void set_last_games_flush_ts(int64_t ts) { last_games_flush_ts_ = ts; }

 private:
  TrainingServerClient(const Params&);
  ~TrainingServerClient();

  void send_handshake();
  void recv_handshake();
  void pause();
  void send_metrics();
  void send_pause_ack();
  void unpause();
  void reload_weights(const std::string& model_filename);
  void loop();
  bool all_pause_notifications_received() const;

  static TrainingServerClient* instance_;

  const int64_t proc_start_ts_;
  const std::string cuda_device_;
  int64_t last_games_flush_ts_ = 0;
  io::Socket* socket_;
  std::thread* thread_;
  std::vector<PauseListener*> pause_listeners_;
  std::vector<ReloadWeightsListener*> reload_weights_listeners_;
  std::vector<MetricsRequestListener*> metrics_request_listeners_;
  int client_id_ = -1;  // assigned by training-server
  int cur_generation_;

  std::condition_variable pause_cv_;
  mutable std::mutex pause_mutex_;
  bool pause_complete_ = false;
};

}  // namespace core

#include <inline/core/TrainingServerClient.inl>
