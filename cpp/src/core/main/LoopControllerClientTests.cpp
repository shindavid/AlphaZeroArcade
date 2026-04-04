#include "core/LoopControllerClient.hpp"
#include "core/MockLoopControllerSocket.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

namespace {

using Socket = core::MockLoopControllerSocket;
using Client = core::LoopControllerClientImpl<Socket>;

// Returns minimal params: only client_role is required
Client::Params make_params(const std::string& role = "self-play-worker") {
  Client::Params params;
  params.client_role = role;
  params.loop_controller_port = 0;  // won't be used (init_for_test bypasses socket creation)
  return params;
}

// Push a valid handshake-ack to the mock socket, so the constructor succeeds.
void push_handshake_ack(Socket& sock, int client_id = 1) {
  boost::json::object ack;
  ack["type"] = "handshake-ack";
  ack["client_id"] = client_id;
  sock.push_incoming(ack);
}

// Push a quit message to the mock socket so that loop() exits and join_loop_thread() returns.
void push_quit(Socket& sock) {
  boost::json::object quit;
  quit["type"] = "quit";
  sock.push_incoming(quit);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture
// ─────────────────────────────────────────────────────────────────────────────

class LoopControllerClientTest : public ::testing::Test {
 protected:
  void TearDown() override {
    mit::reset();
    Client::reset_for_test();
  }

  Socket sock_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Basic handshake tests (constructor; no loop thread)
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(LoopControllerClientTest, HandshakeSuccess) {
  push_handshake_ack(sock_, /*client_id=*/7);

  Client::init_for_test(make_params(), &sock_);
  Client* client = Client::get();

  ASSERT_NE(client, nullptr);
  EXPECT_EQ(client->client_id(), 7);
}

TEST_F(LoopControllerClientTest, HandshakeRejected) {
  boost::json::object ack;
  ack["type"] = "handshake-ack";
  ack["client_id"] = 1;
  ack["rejection"] = "duplicate role";
  sock_.push_incoming(ack);

  EXPECT_THROW(Client::init_for_test(make_params(), &sock_), util::CleanException);
}

// ─────────────────────────────────────────────────────────────────────────────
// Quit → deactivated
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(LoopControllerClientTest, QuitSetsDeactivated) {
  mit::seed(42);

  push_handshake_ack(sock_);
  push_quit(sock_);

  Client::init_for_test(make_params(), &sock_);
  Client* client = Client::get();
  client->start();
  client->join_loop_thread();

  EXPECT_TRUE(Client::deactivated());
}

// ─────────────────────────────────────────────────────────────────────────────
// data-request forwarded to listener
// ─────────────────────────────────────────────────────────────────────────────

class MockDataRequestListener
    : public core::LoopControllerListener<core::LoopControllerInteractionType::kDataRequest> {
 public:
  void handle_data_request(int n_rows, int next_row_limit) override {
    last_n_rows_ = n_rows;
    last_limit_ = next_row_limit;
    called_ = true;
  }
  void handle_data_pre_request(int /*n_rows_limit*/) override {}

  bool called_ = false;
  int last_n_rows_ = -1;
  int last_limit_ = -1;
};

TEST_F(LoopControllerClientTest, DataRequestForwarded) {
  mit::seed(42);

  push_handshake_ack(sock_);

  boost::json::object req;
  req["type"] = "data-request";
  req["n_rows"] = 100;
  req["next_row_limit"] = 200;
  sock_.push_incoming(req);

  push_quit(sock_);

  Client::init_for_test(make_params(), &sock_);
  Client* client = Client::get();

  MockDataRequestListener listener;
  client->add_listener(&listener);

  client->start();
  client->join_loop_thread();

  EXPECT_TRUE(listener.called_);
  EXPECT_EQ(listener.last_n_rows_, 100);
  EXPECT_EQ(listener.last_limit_, 200);
}

// ─────────────────────────────────────────────────────────────────────────────
// reload-weights forwarded to listener
// ─────────────────────────────────────────────────────────────────────────────

class MockReloadWeightsListener
    : public core::LoopControllerListener<core::LoopControllerInteractionType::kReloadWeights> {
 public:
  void reload_weights(const std::vector<char>& buf) override {
    received_buf_ = buf;
    called_ = true;
  }

  bool called_ = false;
  std::vector<char> received_buf_;
};

TEST_F(LoopControllerClientTest, ReloadWeightsForwarded) {
  mit::seed(42);

  push_handshake_ack(sock_);

  // Push file bytes before the message (recv_file_bytes is called right after
  // type=="reload-weights")
  std::vector<char> expected_bytes = {'w', 'e', 'i', 'g', 'h', 't', 's'};
  sock_.push_incoming_file(expected_bytes);

  boost::json::object rw;
  rw["type"] = "reload-weights";
  sock_.push_incoming(rw);

  push_quit(sock_);

  Client::init_for_test(make_params(), &sock_);
  Client* client = Client::get();

  MockReloadWeightsListener listener;
  client->add_listener(&listener);

  client->start();
  client->join_loop_thread();

  EXPECT_TRUE(listener.called_);
  EXPECT_EQ(listener.received_buf_, expected_bytes);
}

// ─────────────────────────────────────────────────────────────────────────────
// pause / unpause ─ pause-ack sent only after all listeners have checked in
// ─────────────────────────────────────────────────────────────────────────────

/*
 * MockPauseListener lets each listener run pause()/unpause() in a separate mit::thread so that
 * the ordering of handle_pause_receipt() calls relative to the loop thread is under the
 * MIT scheduler's control.
 */
class MockPauseListener
    : public core::LoopControllerListener<core::LoopControllerInteractionType::kPause> {
 public:
  explicit MockPauseListener(Client* client, int id) : client_(client), id_(id) {}

  // Called by the loop thread. Spawns a worker thread that eventually calls
  // handle_pause_receipt(). The mit scheduler decides the interleaving.
  void pause() override {
    pause_thread_ = new mit::thread([this]() {
      client_->handle_pause_receipt(__FILE__, __LINE__);
      del_pause_thread_ = true;
    });
  }

  void unpause() override {
    unpause_thread_ = new mit::thread([this]() {
      client_->handle_unpause_receipt(__FILE__, __LINE__);
      del_unpause_thread_ = true;
    });
  }

  void join() {
    if (pause_thread_ && pause_thread_->joinable()) pause_thread_->join();
    if (unpause_thread_ && unpause_thread_->joinable()) unpause_thread_->join();
    delete pause_thread_;
    delete unpause_thread_;
    pause_thread_ = nullptr;
    unpause_thread_ = nullptr;
  }

  int id() const { return id_; }

 private:
  Client* client_;
  int id_;
  mit::thread* pause_thread_ = nullptr;
  mit::thread* unpause_thread_ = nullptr;
  bool del_pause_thread_ = false;
  bool del_unpause_thread_ = false;
};

// Verify that pause-ack and unpause-ack messages appear in the outgoing stream only
// after all listeners have called handle_pause_receipt() / handle_unpause_receipt().
//
// Run with multiple seeds to stress-test the receipt ordering race.
class PauseUnpauseTest : public LoopControllerClientTest {
 protected:
  void run_once(int seed, int n_listeners) {
    mit::reset();
    mit::seed(seed);

    push_handshake_ack(sock_);

    boost::json::object pause_msg;
    pause_msg["type"] = "pause";
    sock_.push_incoming(pause_msg);

    boost::json::object unpause_msg;
    unpause_msg["type"] = "unpause";
    sock_.push_incoming(unpause_msg);

    push_quit(sock_);

    Client::init_for_test(make_params(), &sock_);
    Client* client = Client::get();

    std::vector<MockPauseListener*> listeners;
    for (int i = 0; i < n_listeners; ++i) {
      auto* l = new MockPauseListener(client, i);
      client->add_listener(l);
      listeners.push_back(l);
    }

    client->start();
    client->join_loop_thread();

    for (auto* l : listeners) {
      l->join();
      delete l;
    }

    // Expect: handshake, pause-ack, unpause-ack, done  (in that order)
    // handshake message is sent by the constructor before loop starts.
    // The "done" message is sent by reset_for_test() → shutdown() → send_done(), but since
    // reset_for_test() sets shutdown_initiated_ = true *before* delete, send_done() is skipped.
    // So we expect: handshake, pause-ack, unpause-ack.
    boost::json::value msg1 = sock_.pop_outgoing();
    EXPECT_EQ(msg1.at("type").as_string(), "handshake");

    boost::json::value msg2 = sock_.pop_outgoing();
    EXPECT_EQ(msg2.at("type").as_string(), "pause-ack");

    boost::json::value msg3 = sock_.pop_outgoing();
    EXPECT_EQ(msg3.at("type").as_string(), "unpause-ack");

    EXPECT_EQ(sock_.outgoing_size(), 0);

    Client::reset_for_test();
    // Don't call the fixture TearDown reset again
  }
};

TEST_F(PauseUnpauseTest, SingleListener) {
  for (int seed = 0; seed < 20; ++seed) {
    run_once(seed, /*n_listeners=*/1);
  }
}

TEST_F(PauseUnpauseTest, TwoListeners) {
  for (int seed = 0; seed < 20; ++seed) {
    run_once(seed, /*n_listeners=*/2);
  }
}

TEST_F(PauseUnpauseTest, ThreeListeners) {
  for (int seed = 0; seed < 20; ++seed) {
    run_once(seed, /*n_listeners=*/3);
  }
}

}  // namespace

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
