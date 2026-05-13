#!/usr/bin/env python3
"""
Diagnostic: extract a gen-N.onnx file's `nnue/*` BackupNet weights and check
how the BackupNet behaves on (acc=0, z_s=0) as Qs* and Ws* are swept.

If the BackupLoss anneal has driven the model toward an identity passthrough,
we expect:
    softmax(Q_logits)  -> active-seat win-share approximately equal to Qs*
    W_out              -> approximately equal to Ws*

This script loads weights directly from the ONNX initializers (the same path
the C++ side uses via core::parse_model_bundle), so it tests the bytes the C++
binary actually consumes -- not whatever happens to be in memory after training.

Usage:
    ./py/tools/inspect_backup_net.py PATH_TO/gen-N.onnx
"""
from shared.backup_net import BackupNet

import argparse
import sys
from typing import Dict

import numpy as np
import onnx
from onnx import numpy_helper
import torch


# Must match c4::beta0::Spec::BackupNetDims (cpp/include/games/connect4/Bindings.hpp).
C4_VALUE_DIM = 3  # WLD
C4_BACKUP_DIMS = dict(
    static_latent_dim=4,
    embed_dim=64,
    layer1_dim=32,
    layer2_dim=16,
    action_latent_dim=8,
)


def extract_nnue_initializers(onnx_path: str) -> Dict[str, np.ndarray]:
    """Return all initializers whose name starts with 'nnue/', stripped of that prefix."""
    model = onnx.load(onnx_path)
    out: Dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        if not init.name.startswith('nnue/'):
            continue
        arr = numpy_helper.to_array(init)
        out[init.name[len('nnue/'):]] = arr
    return out


def summarize(name: str, arr: np.ndarray):
    flat = arr.reshape(-1).astype(np.float64)
    print(f'  {name:24s} shape={tuple(arr.shape)!s:24s} '
          f'norm={np.linalg.norm(flat):.5g}  '
          f'min={flat.min():+.4g}  max={flat.max():+.4g}  '
          f'mean={flat.mean():+.4g}')


def build_backup_net_from_weights(weights: Dict[str, np.ndarray]) -> BackupNet:
    net = BackupNet(
        value_dim=C4_VALUE_DIM,
        static_latent_dim=C4_BACKUP_DIMS['static_latent_dim'],
        embed_dim=C4_BACKUP_DIMS['embed_dim'],
        layer1_dim=C4_BACKUP_DIMS['layer1_dim'],
        layer2_dim=C4_BACKUP_DIMS['layer2_dim'],
    )
    state: Dict[str, torch.Tensor] = {}
    for prefix in ('layer1', 'layer2', 'out'):
        state[f'{prefix}.weight'] = torch.from_numpy(
            weights[f'{prefix}.weight'].astype(np.float32))
        state[f'{prefix}.bias'] = torch.from_numpy(
            weights[f'{prefix}.bias'].astype(np.float32))
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def softmax_to_active_win_share(logits: np.ndarray) -> float:
    """WLD logits -> active-seat win-share = p_win + 0.5 * p_draw."""
    shifted = logits - logits.max()
    ex = np.exp(shifted).astype(np.float64)
    p = ex / ex.sum()
    return float(p[0] + 0.5 * p[2])


def sweep(net: BackupNet, label: str, qs_values, ws_values):
    """Run BackupNet on acc=0, z_s=0 across the Cartesian product of (qs, ws)."""
    embed = C4_BACKUP_DIMS['embed_dim']
    d_s = C4_BACKUP_DIMS['static_latent_dim']
    print(f'\n=== Sweep: {label} ===')
    print(f'{"Qs*":>8s} {"Ws*":>8s} | {"Q_active":>10s} {"W_out":>10s} | '
          f'{"Q_logits[W,L,D]":>30s}')
    with torch.no_grad():
        for qs in qs_values:
            for ws in ws_values:
                acc = torch.zeros(1, embed)
                z_s = torch.zeros(1, d_s)
                Qs = torch.tensor([qs], dtype=torch.float32)
                Ws = torch.tensor([ws], dtype=torch.float32)
                out = net(acc, z_s, Qs, Ws)[0].numpy()
                logits = out[:C4_VALUE_DIM]
                w_out = float(out[C4_VALUE_DIM])
                q_active = softmax_to_active_win_share(logits)
                logits_str = ', '.join(f'{x:+.3f}' for x in logits)
                print(f'{qs:8.3f} {ws:8.3f} | {q_active:10.4f} {w_out:10.4f} | '
                      f'[{logits_str}]')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_path', type=str)
    args = parser.parse_args()

    print(f'Loading ONNX: {args.onnx_path}')
    weights = extract_nnue_initializers(args.onnx_path)

    expected_keys = {
        'child_embed.weight', 'child_embed.bias',
        'layer1.weight', 'layer1.bias',
        'layer2.weight', 'layer2.bias',
        'out.weight', 'out.bias',
    }
    missing = expected_keys - set(weights.keys())
    extra = set(weights.keys()) - expected_keys
    print(f'\nNNUE initializers found: {sorted(weights.keys())}')
    if missing:
        print(f'  MISSING: {sorted(missing)}')
    if extra:
        print(f'  EXTRA:   {sorted(extra)}')

    print('\nNNUE tensor summary:')
    for k in sorted(weights.keys()):
        summarize(k, weights[k])

    expected_shapes = {
        'child_embed.weight': (C4_BACKUP_DIMS['embed_dim'],
                               6 + C4_BACKUP_DIMS['action_latent_dim']),
        'child_embed.bias': (C4_BACKUP_DIMS['embed_dim'],),
        'layer1.weight': (C4_BACKUP_DIMS['layer1_dim'],
                          C4_BACKUP_DIMS['embed_dim']
                          + C4_BACKUP_DIMS['static_latent_dim'] + 2),
        'layer1.bias': (C4_BACKUP_DIMS['layer1_dim'],),
        'layer2.weight': (C4_BACKUP_DIMS['layer2_dim'],
                          C4_BACKUP_DIMS['layer1_dim']),
        'layer2.bias': (C4_BACKUP_DIMS['layer2_dim'],),
        'out.weight': (C4_VALUE_DIM + 1, C4_BACKUP_DIMS['layer2_dim']),
        'out.bias': (C4_VALUE_DIM + 1,),
    }
    print('\nShape check:')
    bad = False
    for k, exp in expected_shapes.items():
        got = tuple(weights[k].shape) if k in weights else None
        ok = got == exp
        marker = 'OK' if ok else 'MISMATCH'
        print(f'  {k:24s} got={got!s:24s} expected={exp!s:24s} {marker}')
        if not ok:
            bad = True
    if bad:
        print('\nABORT: shape mismatches found.')
        sys.exit(1)

    net = build_backup_net_from_weights(weights)

    # Identity-passthrough probe: hold acc=0 and z_s=0, sweep Qs* and Ws*.
    qs_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    ws_values = [0.0, 0.05, 0.1, 0.2, 0.5]
    sweep(net, 'acc=0, z_s=0 (full Qs*/Ws* grid)', qs_values, ws_values)

    # Vary just Qs* with Ws* fixed near the value seen in the bug report.
    sweep(net, 'acc=0, z_s=0, Ws* = 0.0113 (vary Qs*)',
          qs_values, [0.0113])

    # Vary just Ws* with Qs* fixed at 0.5 (neutral).
    sweep(net, 'acc=0, z_s=0, Qs* = 0.5 (vary Ws*)',
          [0.5], ws_values)

    # Reproduce the exact (Qs*, Ws*) pair from the bug report.
    sweep(net, 'acc=0, z_s=0 -- exact bug-report inputs',
          [0.5214], [0.0113])


if __name__ == '__main__':
    main()
