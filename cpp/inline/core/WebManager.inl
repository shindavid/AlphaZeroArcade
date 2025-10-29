#include "core/WebManager.hpp"

#include "util/mit/mit.hpp"  // IWYU pragma: keep

namespace core {

template <core::concepts::Game Game>
WebManager<Game>::WebManager() : acceptor_(create_acceptor()), socket_(io_context_) {
  launch_bridge();
  launch_frontend();

  std::cout << "Please open the frontend in your browser at:\n\n"
            << "    http://localhost:5173\n"
            << std::endl;

  thread_ = mit::thread([this]() { response_loop(); });
}

template <core::concepts::Game Game>
inline WebManager<Game>* WebManager<Game>::get_instance() {
  static WebManager<Game> instance;
  return &instance;
}

template <core::concepts::Game Game>
void WebManager<Game>::send_msg(const boost::json::object& msg) {
  std::string out = boost::json::serialize(msg) + "\n";
  boost::asio::write(socket_, boost::asio::buffer(out));
}

template <core::concepts::Game Game>
void WebManager<Game>::launch_bridge() {
  namespace bp = boost::process;
  namespace bf = boost::filesystem;

  os_util::free_port(bridge_port_);

  bp::environment env = boost::this_process::environment();
  env["BRIDGE_PORT"] = std::to_string(bridge_port_);
  env["ENGINE_PORT"] = std::to_string(engine_port_);
  env["SPAWN_ENGINE"] = "false";

  bf::path start_dir = "/workspace/repo/web";
  bf::path log_dir = std::format("/home/devuser/scratch/logs/{}", Game::Constants::kGameName);
  bf::create_directories(log_dir);
  bf::path log_file = log_dir / "bridge.log";

  std::string cmd = "npm run bridge";

  bridge_process_ = new bp::child(cmd, bp::start_dir = start_dir, bp::std_out > log_file,
                                  bp::std_err > log_file, env);

  LOG_INFO("Web player launched bridge process on port {}", bridge_port_);
}

template <core::concepts::Game Game>
void WebManager<Game>::launch_frontend() {
  namespace bp = boost::process;
  namespace bf = boost::filesystem;

  os_util::free_port(vite_port_);

  bp::environment env = boost::this_process::environment();
  env["VITE_BRIDGE_PORT"] = std::to_string(bridge_port_);

  bf::path start_dir = "/workspace/repo/web";
  bf::path log_dir = std::format("/home/devuser/scratch/logs/{}", Game::Constants::kGameName);
  bf::create_directories(log_dir);
  bf::path log_file = log_dir / "frontend.log";

  std::string cmd = std::format("npm --workspace=games/{} run dev", Game::Constants::kGameName);

  frontend_process_ = new bp::child(cmd, bp::start_dir = start_dir, bp::std_out > log_file,
                                    bp::std_err > log_file, env);

  LOG_INFO("Web player launched frontend process");
}

template <core::concepts::Game Game>
boost::asio::ip::tcp::acceptor WebManager<Game>::create_acceptor() {
  os_util::free_port(engine_port_);

  boost::asio::ip::tcp::acceptor acceptor(io_context_);
  acceptor.open(boost::asio::ip::tcp::v4());
  acceptor.set_option(boost::asio::socket_base::reuse_address(true));
  acceptor.bind(boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), engine_port_));
  acceptor.listen();
  return acceptor;
}

template <core::concepts::Game Game>
void WebManager<Game>::response_loop() {
  LOG_INFO("Web player is waiting for a connection on port {}", engine_port_);
  acceptor_.accept(socket_);
  LOG_INFO("Web player connected to client.");

  {
    mit::unique_lock lock(mutex_);
    bridge_connected_ = true;
  }
  cv_.notify_all();
}

template <core::concepts::Game Game>
void WebManager<Game>::wait_for_connection() {
  mit::unique_lock lock(mutex_);
  cv_.wait(lock, [this]() { return bridge_connected_; });
}

template <core::concepts::Game Game>
void WebManager<Game>::wait_for_new_game_ready() {
  mit::unique_lock lock(mutex_);
  cv_.wait(lock, [this]() { return ready_for_new_game_; });
  ready_for_new_game_ = false;
}

}  // namespace core
