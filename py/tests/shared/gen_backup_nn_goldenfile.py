#!/usr/bin/env python3
"""
Generator for the BackupNNEvaluator C++/Python equivalence test.

This script regenerates two checked-in artifacts:

    test_models/beta0_backup_nn_test.onnx
    goldenfiles/beta0_tests/backup_nn_equivalence.json

The C++ test `backup_nn_equivalence_tests` loads the ONNX file, runs
`beta0::BackupNNEvaluator<c4::beta0::Spec>::compute_child_embedding(...)` and
`apply(...)` on the deterministic scenario described in the JSON file, and
compares its outputs against the Python-side reference values stored therein.

Re-run this script whenever:
  * BackupNet / ChildEmbeddingHead architecture changes
  * The Connect4 beta0 spec's BackupNetDims change (kStaticLatentDim, kEmbedDim, ...)
  * The on-disk format of either artifact changes

Usage:
    python py/tests/shared/gen_backup_nn_goldenfile.py

This script is intentionally NOT run as part of `py/run_tests.py`. It is a manual
regeneration tool, like `py/run_tests.py -w` for ordinary goldenfiles.
"""
import json
import os
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = '/workspace/repo'

from shared.basic_types import ShapeInfo, ShapeInfoCollection
from shared.model import Model
from shared.model_config import ModelConfig, ModuleSpec


# ----------------------------------------------------------------------------
# Architecture constants. These MUST match cpp/include/games/connect4/Bindings.hpp
# (Spec::BackupNetDims) and the c4 game's policy/value shapes. If you change them
# here, also change them in Bindings.hpp (or vice versa).
# ----------------------------------------------------------------------------
SEED = 20260511

C4_INPUT_SHAPE = (2, 6, 7)         # c4::Game InputEncoder shape
C4_NUM_ACTIONS = 7                 # c4::Game policy width
C4_VALUE_DIM = 3                   # WLD logits

# BackupNet dims (must match c4::beta0::Spec::BackupNetDims).
STATIC_LATENT_DIM = 4
EMBED_DIM = 64
BACKUP_LAYER1_DIM = 32
BACKUP_LAYER2_DIM = 16
ZA_DIM = 8

# Trivial trunk to keep the on-disk ONNX small. The main NN's correctness is
# irrelevant to this test — only the BackupNet/ChildEmbeddingHead weights matter.
C_TRUNK = 4

# Test scenario knobs.
NUM_ROUNDS = 5

# Output paths.
ONNX_PATH = os.path.join(REPO_ROOT, 'test_models', 'beta0_backup_nn_test.onnx')
JSON_PATH = os.path.join(
    REPO_ROOT, 'goldenfiles', 'beta0_tests', 'backup_nn_equivalence.json')


def build_model() -> tuple[Model, ShapeInfoCollection]:
    """Build a c4-shaped beta0 model with a trivial trunk + real BackupNet/heads."""
    trunk_shape = (C_TRUNK, *C4_INPUT_SHAPE[1:])
    A = C4_NUM_ACTIONS
    av_dim = 2  # WinShareActionValueHead per-action width
    au_dim = 1
    action_latent_shape = (A, ZA_DIM)

    config = ModelConfig.create(
        external_inputs=['value_baseline', 'value_uncertainty_baseline', 'child_stats'],
        stem=ModuleSpec(type='ConvBlock', args=[C4_INPUT_SHAPE, trunk_shape]),
        policy=ModuleSpec(
            type='PolicyHead',
            args=[trunk_shape, 1, (A,)], parents=['stem']),
        value=ModuleSpec(
            type='WinLossDrawValueHead',
            args=[trunk_shape, 1, 4], parents=['stem']),
        action_value=ModuleSpec(
            type='WinShareActionValueHead',
            args=[trunk_shape, 1, (A, av_dim)], parents=['stem']),
        action_value_uncertainty=ModuleSpec(
            type='ActionValueUncertaintyHead',
            args=[trunk_shape, 1, (A, au_dim)], parents=['stem']),
        static_latent=ModuleSpec(
            type='StaticLatentHead',
            args=[trunk_shape, 2, STATIC_LATENT_DIM], parents=['stem']),
        action_latent=ModuleSpec(
            type='ActionLatentHead',
            args=[trunk_shape, 2, action_latent_shape], parents=['stem']),
        child_embedding=ModuleSpec(
            type='ChildEmbeddingHead',
            args=[(A, 6), action_latent_shape, EMBED_DIM],
            parents=['child_stats', 'action_latent']),
        accumulator=ModuleSpec(
            type='AccumulatorHead', parents=['child_embedding']),
        backup_net=ModuleSpec(
            type='BackupNet',
            kwargs={
                'value_dim': C4_VALUE_DIM,
                'static_latent_dim': STATIC_LATENT_DIM,
                'embed_dim': EMBED_DIM,
                'layer1_dim': BACKUP_LAYER1_DIM,
                'layer2_dim': BACKUP_LAYER2_DIM,
            },
            parents=['accumulator', 'static_latent', 'value_baseline', 'value_uncertainty_baseline']),
    )
    model = Model(config)

    shape_info = ShapeInfoCollection(
        input_shapes={
            'input': ShapeInfo('input', 0, C4_INPUT_SHAPE),
            'child_stats': ShapeInfo('child_stats', 1, (C4_NUM_ACTIONS, 6)),
            'value_baseline': ShapeInfo('value_baseline', 2, (C4_VALUE_DIM,)),
            'value_uncertainty_baseline': ShapeInfo('value_uncertainty_baseline', 3, (1,)),
        },
        target_shapes={},
        head_shapes={
            'policy': ShapeInfo('policy', 0, (C4_NUM_ACTIONS,)),
            'value': ShapeInfo('value', 1, (C4_VALUE_DIM,)),
            'action_value': ShapeInfo('action_value', 2, (C4_NUM_ACTIONS, 2)),
            'action_value_uncertainty': ShapeInfo(
                'action_value_uncertainty', 3, (C4_NUM_ACTIONS, 1)),
            'static_latent': ShapeInfo('static_latent', 4, (STATIC_LATENT_DIM,)),
            'action_latent': ShapeInfo('action_latent', 5, action_latent_shape),
            'child_embedding': ShapeInfo(
                'child_embedding', 6, (C4_NUM_ACTIONS, EMBED_DIM)),
            'accumulator': ShapeInfo('accumulator', 7, (EMBED_DIM,)),
        },
    )
    return model, shape_info


def make_scenario(rng: np.random.Generator) -> Dict:
    """
    Build a deterministic test scenario:
      * Static per-parent inputs: z_s, S_baseline, Ws_baseline
      * Per-action z_a (length C4_NUM_ACTIONS)
      * NUM_ROUNDS snapshots of per-action child_stats
        - Round 0: all zeros except policy P (uniform across all 7 actions)
        - Subsequent rounds simulate plausible MCTS visits, perturbing Qs/Ws/N/AVs/AUs
        - Round 1->2: action 0 stays unchanged (exercises the no-op subtract-add path)
    """
    z_s = rng.standard_normal(STATIC_LATENT_DIM).astype(np.float32)
    z_a = rng.standard_normal((C4_NUM_ACTIONS, ZA_DIM)).astype(np.float32)
    # S_baseline: a normalized WLD distribution in the active-seat-rotated frame.
    raw = rng.standard_normal(C4_VALUE_DIM).astype(np.float32)
    ex = np.exp(raw - raw.max())
    S_baseline = (ex / ex.sum()).astype(np.float32)
    Ws_baseline = float(rng.uniform(0.0, 1.0))

    P = np.full(C4_NUM_ACTIONS, 1.0 / C4_NUM_ACTIONS, dtype=np.float32)

    rounds: List[np.ndarray] = []
    # Round 0: all zeros except P.
    cs0 = np.zeros((C4_NUM_ACTIONS, 6), dtype=np.float32)
    cs0[:, 3] = P  # P_INDEX = 3
    rounds.append(cs0)

    prev = cs0
    for r in range(1, NUM_ROUNDS):
        cur = prev.copy()
        for i in range(C4_NUM_ACTIONS):
            # Skip action 0 between round 1 and round 2 (exercise no-op subtract-add).
            if r == 2 and i == 0:
                continue
            cur[i, 0] = float(rng.uniform(-1.0, 1.0))   # Qs
            cur[i, 1] = float(rng.uniform(0.0, 1.0))    # Ws
            cur[i, 2] = float(rng.integers(0, 5 * r))   # N
            # P stays fixed (cur[i, 3] inherits from prev)
            cur[i, 4] = float(rng.uniform(-1.0, 1.0))   # AVs
            cur[i, 5] = float(rng.uniform(0.0, 1.0))    # AUs
        rounds.append(cur)
        prev = cur

    return {
        'z_s': z_s,
        'z_a': z_a,
        'value_baseline': S_baseline,
        'value_uncertainty_baseline': Ws_baseline,
        'rounds': rounds,
    }


def python_compute_references(model: Model, scenario: Dict) -> Dict:
    """
    Run ChildEmbeddingHead.forward + sum-pool + BackupNet.forward, then return per-round
    embeddings, accumulator, the active-seat-rotated WLD softmax distribution `expected_S`,
    and the W scalar.
    """
    child_emb = model._module_dict['child_embedding']
    backup = model._module_dict['backup_net']

    z_s = scenario['z_s']
    z_a = scenario['z_a']
    S_baseline = scenario['value_baseline']
    Ws_baseline = scenario['value_uncertainty_baseline']

    z_s_t = torch.from_numpy(z_s).unsqueeze(0)                # (1, d_s)
    z_a_t = torch.from_numpy(z_a).unsqueeze(0)                # (1, A, za)
    Ss_t = torch.from_numpy(S_baseline).unsqueeze(0)             # (1, value_dim)
    Ws_t = torch.tensor([[Ws_baseline]], dtype=torch.float32)

    rounds_out = []
    with torch.no_grad():
        for cs in scenario['rounds']:
            cs_t = torch.from_numpy(cs).unsqueeze(0)          # (1, A, 6)
            # Per-action embeddings (1, A, embed_dim).
            e = child_emb(cs_t, z_a_t)
            embeddings = e[0].cpu().numpy().astype(np.float32)  # (A, embed_dim)
            acc_t = e.sum(dim=1)                              # (1, embed_dim)
            accumulator = acc_t[0].cpu().numpy().astype(np.float32)
            # BackupNet: returns (1, value_dim + 1) = (1, 4).
            out = backup(acc_t, z_s_t, Ss_t, Ws_t.squeeze(-1))
            out_np = out[0].cpu().numpy().astype(np.float32)
            value_logits = out_np[:C4_VALUE_DIM]
            W_scalar = float(out_np[C4_VALUE_DIM])
            # Softmax → WLD probabilities (active-seat-rotated frame).
            shifted = value_logits - np.max(value_logits)
            ex = np.exp(shifted).astype(np.float64)
            probs = (ex / ex.sum()).astype(np.float32)
            rounds_out.append({
                'child_stats': cs.tolist(),
                'expected_embeddings': embeddings.tolist(),
                'expected_accumulator': accumulator.tolist(),
                'expected_S': probs.tolist(),
                'expected_W': W_scalar,
            })

    return {
        # Static scenario inputs.
        'static_latent_dim': STATIC_LATENT_DIM,
        'embed_dim': EMBED_DIM,
        'za_dim': ZA_DIM,
        'value_dim': C4_VALUE_DIM,
        'num_actions': C4_NUM_ACTIONS,
        'z_s': scenario['z_s'].tolist(),
        'z_a': scenario['z_a'].tolist(),
        'value_baseline': S_baseline.tolist(),
        'value_uncertainty_baseline': Ws_baseline,
        'rounds': rounds_out,
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print('Building Model...')
    model, shape_info = build_model()
    model.cpu().eval()

    print(f'Saving ONNX -> {ONNX_PATH}')
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
    model.save_model(ONNX_PATH, shape_info)
    onnx_size = os.path.getsize(ONNX_PATH)
    print(f'  ONNX file size: {onnx_size / 1024:.1f} KB')

    print('Building scenario + computing Python references...')
    scenario = make_scenario(rng)
    refs = python_compute_references(model, scenario)

    print(f'Writing goldenfile -> {JSON_PATH}')
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, 'w') as f:
        # Use a generous float repr (~17 digits) so round-trip is exact for fp64;
        # the C++ side compares against these as fp32 with tol 1e-5.
        json.dump(refs, f, indent=2)
        f.write('\n')

    print('Done.')


if __name__ == '__main__':
    main()
