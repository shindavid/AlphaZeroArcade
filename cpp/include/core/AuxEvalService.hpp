#pragma once

namespace core {

struct ModelBundle;

/*
 * Bare-bones polymorphic base for an auxiliary evaluation service that an
 * NNEvaluationService may own alongside its primary GPU-backed neural-network engine.
 *
 * Concrete subclasses (e.g. beta0::BackupNNEvaluator) hold their own state and override
 * reload_weights() to absorb fresh weights from each ModelBundle.
 *
 * Intended usage: the NNEvaluationService accepts an AuxFactory callback at construction
 * time. The factory is invoked exactly once and produces a default-constructed aux service
 * (i.e. no weights loaded yet). The NNEvaluationService then drives the aux service's
 * reload_weights() at the same points it drives its own net's weight loads — once at
 * construction (if a local model file was provided) and once for every subsequent
 * loop-controller-pushed model. Aux services do NOT register themselves as
 * LoopControllerListeners; their lifecycle is owned end-to-end by the NNEvaluationService.
 */
class AuxEvalService {
 public:
  virtual ~AuxEvalService() = default;

  virtual void reload_weights(const core::ModelBundle& model) = 0;
};

}  // namespace core
