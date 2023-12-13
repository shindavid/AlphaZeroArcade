#pragma once

namespace mcts {

/*
 * Responsible for keeping track of all NNEvaluationService instances. This is used to keep track of
 * the number of instances constructed, and to assign a unique instance_id to each instance. These
 * values are attached to messages sent from NNEvaluationService instances to the cmd-server
 * ("response 2 of 5"), so that the cmd-server can determine when a response has been received from
 * each instance.
 */
class NNEvaluationServiceProvider {
 public:
  static int register_instance() { return instance_count_++; }
  static int num_instances() { return instance_count_; }

 private:
  static int instance_count_;
};

}  // namespace mcts
