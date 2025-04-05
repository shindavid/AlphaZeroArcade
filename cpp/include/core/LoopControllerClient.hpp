#pragma once

#include <core/LoopControllerListener.hpp>
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
 * LoopControllerClient is used to communicate with an external loop controller. It is to be used as
 * a singleton, and can further forward messages to any number of listeners.
 *
 * For now, all messages will be in json format. We can revisit this in the future.
 *
 * Usage:
 *
 * core::LoopControllerClient::Params params;
 * // set params
 * core::LoopControllerClient::init(params);
 *
 * core::LoopControllerClient* client = core::LoopControllerClient::get();
 * client->send("abc", 3);
 *
 * To listen to messages from the loop-controller, implement the LoopControllerListener interface
 * and subscribe to the client via client->add_listener(listener).
 */
class LoopControllerClient {
 public:
  struct Params {
    auto make_options_description();
    bool operator==(const Params& other) const = default;

    std::string loop_controller_hostname = "localhost";
    io::port_t loop_controller_port = 0;
    std::string client_role;  // must be specified if port is specified
    std::string ratings_tag;
    std::string cuda_device = "cuda:0";
    std::string output_base_dir;  // needed for direct-game-log-write optimization
    int manager_id = -1;
    int weights_request_generation = -1;
    bool report_metrics = true;
  };

  using PauseListener = LoopControllerListener<LoopControllerInteractionType::kPause>;
  using ReloadWeightsListener = LoopControllerListener<LoopControllerInteractionType::kReloadWeights>;
  using MetricsRequestListener = LoopControllerListener<LoopControllerInteractionType::kMetricsRequest>;
  using DataRequestListener = LoopControllerListener<LoopControllerInteractionType::kDataRequest>;

  static void init(const Params&);
  static bool initialized() { return instance_;  }
  static bool deactivated() { return instance_ && instance_->deactivated_; }
  static LoopControllerClient* get() { return instance_;  }
  int client_id() const { return client_id_; }
  int cur_generation() const { return cur_generation_; }
  const std::string& role() const { return params_.client_role; }
  const std::string& cuda_device() const { return params_.cuda_device; }
  const std::string& ratings_tag() const { return params_.ratings_tag; }
  const std::string& output_base_dir() const { return params_.output_base_dir; }
  bool report_metrics() const { return params_.report_metrics; }
  bool is_loop_controller_local() const {  return params_.loop_controller_hostname == "localhost"; }

  template <typename T>
  void add_listener(T* listener);

  void start();
  void shutdown();
  void send_done();
  void send_with_file(const boost::json::value& msg, const std::vector<char>& buf);
  void send(const boost::json::value& msg) { socket_->json_write(msg); }

  void send_worker_ready();
  void handle_pause_receipt();
  void handle_unpause_receipt();
  PerfStats get_perf_stats() const;

 private:
  LoopControllerClient(const Params&);
  ~LoopControllerClient();

  void send_handshake();
  void recv_handshake();
  void send_pause_ack();
  void send_unpause_ack();
  void pause();
  void unpause();
  void reload_weights(const std::vector<char>& buf, const std::string& cuda_device);
  void handle_data_request(int n_rows);
  void handle_data_pre_request(int n_rows_limit);
  void wait_for_pause_receipts();
  void wait_for_unpause_receipts();
  void loop();

  static LoopControllerClient* instance_;

  const Params params_;
  const int64_t proc_start_ts_;
  io::Socket* socket_;
  std::thread* thread_ = nullptr;
  std::vector<PauseListener*> pause_listeners_;
  std::vector<ReloadWeightsListener*> reload_weights_listeners_;
  std::vector<MetricsRequestListener*> metrics_request_listeners_;
  std::vector<DataRequestListener*> data_request_listeners_;
  int client_id_ = -1;  // assigned by loop-controller
  int cur_generation_ = 0;

  std::condition_variable receipt_cv_;
  mutable std::mutex receipt_mutex_;
  size_t pause_receipt_count_ = 0;
  size_t unpause_receipt_count_ = 0;
  bool shutdown_initiated_ = false;
  bool deactivated_ = false;
};

}  // namespace core

#include <inline/core/LoopControllerClient.inl>
