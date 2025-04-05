#!/usr/bin/env python3

import argparse
from cffi import FFI

import torch
import numpy as np

from alphazero.logic.build_params import BuildParams
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
from util.py_util import CustomHelpFormatter

torch.set_printoptions(linewidth=120, sci_mode=False)
np.set_printoptions(precision=4, suppress=True, linewidth=80)

def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    BuildParams.add_args(parser, add_binary_path_option=False)
    return parser.parse_args()

def build_ffi(run_params, build_params):
    ffi = FFI()
    print(f'ffi lib path: {build_params.get_ffi_lib_path(run_params.game)}')
    lib = ffi.dlopen(build_params.get_ffi_lib_path(run_params.game))

    ffi.cdef("""
        struct State {};
        struct State* Game_new_state(int stones_left, int mode);
        void Game_delete_state(struct State* state);
        struct GameTensor {};
        void Game_tensorize(struct State* start_state, int num_states, float* input_values);
        struct PerfectStrategy {};
        struct PerfectStrategy* PerfectStrategy_new();
        void PerfectStrategy_delete(struct PerfectStrategy* strategy);
        float get_state_value_before(struct PerfectStrategy* strategy, int stones_left);
        float get_state_value_after(struct PerfectStrategy* strategy, int stones_left);
        int get_optimal_action(struct PerfectStrategy* strategy, int stones_left);
    """)
    return ffi, lib

def get_input_tensor(ffi, lib):
    tensor_list = []
    state_list = []
    kMaxStonesToTake = 21
    tensor_dim = (1, 6, 1)
    for stones_left in range(kMaxStonesToTake, 0, -1):
        for mode in (0, 1):
            state = lib.Game_new_state(stones_left, mode)
            input_tensor = torch.empty(tensor_dim, dtype=torch.float32)
            input_values_c = ffi.cast('float*', input_tensor.data_ptr())
            lib.Game_tensorize(state, 1, input_values_c)
            tensor_list.append(input_tensor)
            state_list.append((stones_left, mode))
            lib.Game_delete_state(state)

    input_tensor = torch.stack(tensor_list, axis=0)
    return state_list, input_tensor

def get_perfect_strategy_V(state_list, strategy, lib):
    V = []
    for stones_left, mode in state_list:
        if mode == 0:
            v = lib.get_state_value_before(strategy, stones_left)
        else:
            v = lib.get_state_value_after(strategy, stones_left)
        V.append(v)
    return V

def get_perfect_strategy_AV(state_list, strategy, lib):
    AV = []
    for stones_left, mode in state_list:
        if mode == 1:
            av = np.array([1 - lib.get_state_value_before(strategy, stones_left - i) \
                           if i <= stones_left else 0 \
                           for i in range(0, 3)])
        else:
            av = np.array([lib.get_state_value_after(strategy, stones_left - i)
                           if i <= stones_left else 0 \
                           for i in range(1, 4)])
        AV.append(av)
    return np.stack(AV, axis=0)

def nn_eval_gen(input_tensor, gen, organizer):
    model_path = organizer.get_model_filename(gen)
    model = torch.jit.load(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)
    policy, value, action_values = model(input_tensor)
    policy = torch.softmax(policy, dim=-1).cpu().detach()
    value = torch.softmax(value, dim=-1).cpu().detach()
    action_values = torch.sigmoid(action_values).cpu().detach()
    return policy, value, action_values

def calc_target_av(state_list, V_dict):
    AV = []
    for stones_left, mode in state_list:
        if mode == 1:
            av = np.array([1 - V_dict[(stones_left - i, 0)] \
                           if i < stones_left else 0 \
                           for i in range(0, 3)])
        else:
            av = np.array([V_dict[(stones_left - i, 1)]
                           if i < stones_left else 0 \
                           for i in range(1, 4)])
        AV.append(av)
    return np.stack(AV, axis=0)

def main():
    args = load_args()
    run_params = RunParams.create(args, require_tag=False)
    build_params = BuildParams.create(args)
    organizer = DirectoryOrganizer(run_params)


    ffi, lib = build_ffi(run_params, build_params)
    state_list, input_tensor = get_input_tensor(ffi, lib)
    strategy = lib.PerfectStrategy_new()
    V = get_perfect_strategy_V(state_list, strategy, lib)
    AV = get_perfect_strategy_AV(state_list, strategy, lib)

    gen = 700
    policy, value, action_values = nn_eval_gen(input_tensor, gen, organizer)
    value_dict = dict(zip(state_list, value[:, 0].numpy()))
    target_av = calc_target_av(state_list, value_dict)
    error_V = torch.absolute(value[:, 0] - torch.tensor(V, dtype=torch.float32))
    sv = sorted(zip(state_list, error_V), key=lambda x: x[1], reverse=True)

    for state, nn_v, perfect_v in zip(state_list, value, V):
        print(f'State: {state}, NN V: {nn_v.numpy()[0]}, Perfect V: {perfect_v}, error: {np.absolute(nn_v.numpy()[0] - perfect_v)}')

    for state, av, perfect_av in zip(state_list, action_values, torch.tensor(AV, dtype=torch.float32)):
        print(f'State: {state}, NN AV: {av}, Perfect AV: {perfect_av}, error: {torch.linalg.norm(av - perfect_av)}')

    for state, av, tav in zip(state_list, action_values, torch.tensor(target_av, dtype=torch.float32)):
        print(f'State: {state}, NN AV: {av}, Target AV: {tav}, error: {torch.linalg.norm(av - tav)}')

    for (stones_left, mode), nn_policy in zip(state_list, policy):
        if mode == 1:
            continue
        perfect_action = lib.get_optimal_action(strategy, stones_left)
        nn_action = np.argmax(nn_policy.numpy())

        print(f'State: {stones_left}, NN prior: {nn_policy}, NN: {nn_action}, Optimal action: {perfect_action}, wrong: {nn_action != perfect_action}')

if __name__ == '__main__':
    main()
