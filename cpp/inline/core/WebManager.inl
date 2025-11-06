#include "core/WebManager.hpp"

#include "util/LoggingUtil.hpp"
#include "util/OsUtil.hpp"

#include <format>

namespace core {

template <core::concepts::Game Game>
WebManager<Game>::WebManager() : acceptor_(create_acceptor()), socket_(io_context_) {
  launch_bridge();
  launch_frontend();

  std::cout << "Please open the frontend in your browser at:\n\n"
            << std::format("    http://localhost:{}\n", vite_port_)
            << std::endl;

  thread_ = mit::thread([this]() { response_loop(); });
}

template <core::concepts::Game Game>
WebManager<Game>::~WebManager() {
  if (bridge_process_) {
    bridge_process_->terminate();
    delete bridge_process_;
  }
  if (frontend_process_) {
    frontend_process_->terminate();
    delete frontend_process_;
  }

  {
    mit::unique_lock lock(mutex_);
    bridge_connected_ = false;
    cv_.notify_all();
  }
  if (thread_.joinable()) {
    thread_.join();
  }
}

template <core::concepts::Game Game>
inline WebManager<Game>* WebManager<Game>::get_instance() {
  static WebManager<Game> instance;
  return &instance;
}

template <core::concepts::Game Game>
void WebManager<Game>::send_msg(const boost::json::object& msg) {
  mit::unique_lock lock(io_mutex_);
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
void WebManager<Game>::register_client(WebManagerClient* client) {
  mit::unique_lock lock(mutex_);
  for (seat_index_t seat = 0; seat < Game::Constants::kNumPlayers; ++seat) {
    if (clients_[seat] == nullptr) {
      clients_[seat] = client;
      return;
    }
  }
  throw util::Exception("All seats are already registered with clients");
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

  while (bridge_connected_) {
    try {
      boost::asio::streambuf buf;
      boost::asio::read_until(socket_, buf, "\n");
      std::istream is(&buf);
      std::string line;
      std::getline(is, line);
      auto parsed = boost::json::parse(line);
      const auto& obj = parsed.as_object();

      std::string msg_type = obj.at("type").as_string().c_str();

      if (msg_type == "new_game") {
        for (auto& client : clients_) {
          if (client) {
            client->handle_start_game();
          }
        }
      } else {
        int seat_index = boost::json::value_to<int>(obj.at("seat"));
        if (msg_type == "make_move") {
          boost::json::object payload = obj.at("payload").as_object();
          client_at_seat(seat_index)->handle_action(payload);
        } else if (msg_type == "resign") {
          client_at_seat(seat_index)->handle_resign();
        } else {
          throw util::Exception("Unknown message type: {}", msg_type);
        }
      }
    } catch (const std::exception& ex) {
      LOG_INFO("WebManager: connection closed or error: {}", ex.what());
      bridge_connected_ = false;
      cv_.notify_all();
      break;
    }
  }
}

template <core::concepts::Game Game>
void WebManager<Game>::wait_for_connection() {
  mit::unique_lock lock(mutex_);
  cv_.wait(lock, [this]() { return bridge_connected_; });
}

template <core::concepts::Game Game>
WebManagerClient* WebManager<Game>::client_at_seat(seat_index_t seat) {
  mit::unique_lock lock(mutex_);
  for (auto* c : clients_) {
    if (c && c->seat() == seat) {
      return c;
    }
  }
  throw util::Exception("No client registered at seat {}", seat);
}

}  // namespace core
