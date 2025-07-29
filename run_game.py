#!/usr/bin/env python3
import argparse
import contextlib
import os
import socket
import subprocess
import sys
import time

def get_args():
    parser = argparse.ArgumentParser(description="Run the Tic-Tac-Toe game with a C++ engine and Node.js bridge.")
    parser.add_argument('--rebuild', action='store_true', help="Rebuild the C++ engine before starting")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Utility: find a free TCP port on localhost
# -----------------------------------------------------------------------------
def find_port():
    with contextlib.closing(socket.socket()) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# -----------------------------------------------------------------------------
# Build engine
# -----------------------------------------------------------------------------
def build_engine():
    # Ensure the C++ engine is built before running
    print("Building C++ engine...")
    cmd = ['g++', 'cpp/mock_tictactoe.cpp', '-O3', '-std=c++23', '-o', 'mock_tictactoe']
    subprocess.run(cmd, check=True)
    print("C++ engine built successfully.")

# -----------------------------------------------------------------------------
# Launch the Node.js WebSocket bridge
# -----------------------------------------------------------------------------
def launch_bridge(bridge_port, engine_port):
    env = os.environ.copy()
    env.update({
        'BRIDGE_PORT': str(bridge_port),
        'ENGINE_PORT': str(engine_port),
        'SPAWN_ENGINE': 'false',  # our bridge does _not_ spawn the engine
    })
    return subprocess.Popen(
        ['npm', 'start'],
        cwd='web/server',
        env=env,
        stdout=open('bridge.log', 'w'),
        stderr=subprocess.STDOUT
    )

# -----------------------------------------------------------------------------
# Launch the React frontend (Vite dev server)
# -----------------------------------------------------------------------------
def launch_frontend(bridge_port):
    # Clone the environment and inject the Vite‐runtime var
    env = os.environ.copy()
    env['VITE_BRIDGE_PORT'] = str(bridge_port)

    # Spawn Vite directly in the Tic‑Tac‑Toe workspace
    return subprocess.Popen(
        ['npm', 'run', 'dev'],
        cwd=os.path.join('web', 'games', 'tictactoe'),
        env=env,
        stdout=open('frontend.log', 'w'),
        stderr=subprocess.STDOUT
    )

# -----------------------------------------------------------------------------
# Launch the C++ engine (mock or real)
# -----------------------------------------------------------------------------
def launch_engine(engine_port):
    return subprocess.Popen(
        ['./mock_tictactoe', '--port', str(engine_port)],
        stdout=open('engine.log', 'w'),
        stderr=subprocess.STDOUT
    )

# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
def main():
    args = get_args()

    bridge_port = find_port()
    engine_port = find_port()

    # 0) Build engine
    if args.rebuild:
        build_engine()

    print(f"Using bridge port {bridge_port}, engine port {engine_port}")

    # 1) Start the engine first so its TCP port is bound
    p_engine = launch_engine(engine_port)
    time.sleep(1)  # give the engine a moment to bind and accept

    # 2) Now start the bridge (which will successfully connect to the engine)
    p_bridge = launch_bridge(bridge_port, engine_port)
    time.sleep(1)

    # 3) Finally start the frontend
    p_front = launch_frontend(bridge_port)

    # 4) Wait for the engine to exit; if it crashes, shut down the others
    code = p_engine.wait()
    if code != 0:
        print(f"⚠️  Engine exited with code {code}, shutting down bridge & frontend.")
        p_bridge.terminate()
        p_front.terminate()
        sys.exit(code)

    # 5) Clean exit: stop bridge & frontend too
    p_bridge.terminate()
    p_front.terminate()
    sys.exit(0)


if __name__ == '__main__':
    main()
