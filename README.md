
# Alpha Zero Arcade

A generic AlphaZero framework.

There are many AlphaZero implementations out there, but most of them are for a specific game. Some implementations are
more generic but at a serious performance cost. This implementation is designed to be maximally generic at minimal
overhead.

The implementation aims to incorporate as many state-of-the-art ideas and techniques as possible from other projects.
In particular, it borrows heavily from [KataGo](https://github.com/lightvector/KataGo). Eventually, it hopes to work
just-as-well as KataGo does for go, while minimizing go-specific details in its implementation.

The framework also aims to support games that have one or more of the following characteristics:

* Imperfect information
* Randomness
* `p` players for any `p >= 1`

## Getting Started

### Requirements

To run this project successfully, please ensure your system meets the following requirements:

1. **A Computer with an NVIDIA GPU**: We use CUDA 12.x, which requires an NVIDIA GPU with Compute Capability 7.0 (Volta architecture) or higher. Check compatibility for your GPU [here](https://developer.nvidia.com/cuda-gpus).

2. **NVIDIA GPU Driver**: We use CUDA 12.x, which requires **version 535.54.03** or newer. See the [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for details.

3. **Docker**: See [installation instructions](https://docs.docker.com/engine/install/).

4. **NVIDIA Container Toolkit**: Follow the [installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), including the "Configuring Docker" section.

5. **Python3**: During setup, only one third-party dependency is needed: the `packaging` module, which can be installed via `pip`.

### Initial Setup

To get started, clone the repo, and then run:

```
$ ./setup_wizard.py
```

This will walk you through initial setup, including validation that you have met the above requirements. Part of the
setup involves downloading the latest `AlphaZeroArcade` Docker image from [docker.io](https://hub.docker.com/r/dshin83/alphazeroarcade/tags).

After completing this setup, you will be able to run a Docker container via:

```
$ ./run_docker.py
```

Your first execution of this will issue a `docker run` call to spawn a container instance running `bash`, with your repo-checkout-directory
mounted to `/workspace/`. Subsequent executions of this will issue a `docker exec` call that executes into that
container instance.

You can think of each bash session spawned by `./run_docker.py` as a sort of ssh-session into a virtual machine.
All the work you do will be within this virtual machine.

If you have access to multiple machines, you can launch `run_docker.py` on each of them, and then launch commands to
effectively perform a big distributed AlphaZero run.

### Building

Within your docker container, from the `/workspace/repo/` directory, run:

```
./py/build.py
```

This should build a binary for each supported game, along with some unit-test binaries. The end of the output should list
the built targets:

```
...
target/Release/bin/othello
target/Release/bin/tictactoe
target/Release/bin/tests/blokus_tests
target/Release/bin/tests/c4_tests
...
```
You can then run for example `target/Release/bin/tictactoe -h` to get a list of help options.

### VSCode

Once you have a Docker container running via `./run_docker.py`, the best workflow is to have your IDE connect to it.

For VSCode, this can be accomplished as follows:

1. Install the "Remote Development" extension.
2. Launch the Command Palette with Ctrl+Shift+P, and search for "Attach to Running Container".
3. Select the container

After you've built the project at least once, you will have a `target/` directory that IntelliSense can use to streamline
your coding experience. To set up IntelliSense:

1. Install the "C/C++" extension.
2. From the repo-root of the host-machine, in `.vscode/c_cpp_properties.json`, add a configuration for "AlphaZeroArcade":
```
{
  "configurations": [
    {
      "name": "Linux",
      ...
    },
    {
      "name": "AlphaZeroArcade",
      "compileCommands": "${workspaceFolder}/target/Release/compile_commands.json"
    }
  ],
  "version": 4
}
```
3. Select the New Configuration in VSCode:
- Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on macOS).
- Type "C/C++: Select a Configuration" and select your newly added "AlphaZeroArcade" configuration.

_(Alternatively, you can click the C/C++ configuration indicator in the lower status bar and choose the "AlphaZeroArcade" configuration if it’s visible.)_

### The AlphaZero loop

A highly experienced graphic artist was hired to create the below figure, which summarizes the server architecture:

![image](https://github.com/shindavid/AlphaZeroArcade/assets/5217927/8349eaee-6cc2-4ad7-9fca-dcc8db973ab7)

The loop controller manages the entire loop. It uses its GPU to continuously train a neural network. It periodically
sends snapshots of the network weights over TCP to one or more self-play servers. The self-play servers use the
current network weights to generate self-play games, sending them back to the loop controller over TCP. The loop
controller in return writes those games to disk, along with associated metadata to an sqlite3 database, and
incorporates that data into its continuous training.

One or more ratings servers can also participate. The loop controller sends model weights to the ratings servers,
along with requests to test those weights in specific matchups against a game-specific family of reference agents.
The ratings servers performs these matches and then send back match results (win/loss/draw counts). The loop controller
writes the results to its database, and computes ratings based on the results. These ratings, along with various
self-play and training metrics, can be viewed using the web dashboard.

You can launch this loop on your local machine for the game of your choice, with a command like this:

```
./py/alphazero/scripts/run_local.py --game c4 --tag my-first-run
```
or using aliases,
```
./py/alphazero/scripts/run_local.py -g c4 -t my-first-run
```
This launches one instance of each the 3 server types (loop-controller, self-play, ratings).

Here `my-first-run` is a run _tag_. All files produced by the run will then be placed in the directory

```
/workspace/output/c4/my-first-run/
```

By default, `run_local.py` will detect the number of available cuda devices on the local machine, and allocate
the GPU's across the servers in an optimal manner. If 1 or more GPU's are shared by multiple servers, the
loop-controller carefully orchestrates the entire process, pausing and unpausing components as needed to ensure
that the GPU's stay fully utilized, without the components thrashing with each other or getting starved indefinitely.

### Measuring Progress

During-or-after a run of the loop-controller, you can launch a web dashboard to track the progress of your run:

```
./py/alphazero/scripts/launch_dashboard.py -g c4 -t my-first-run
```

This will print a URL that you can paste into a web browser on your local machine, which currently looks like this:

![image](https://github.com/shindavid/AlphaZeroArcade/assets/5217927/663d1585-5fdd-4a5f-bcae-91d211559466)

The "Ratings" item in the sidebar shows a plot like this:

![image](https://github.com/shindavid/AlphaZeroArcade/assets/5217927/a8c1edb8-425e-4634-803f-086801aa59cd)

The dark curve corresponds to an MCTS agent using i=1600 iterations per search. The light curve corresponds to an agent
that plays according to the raw network policy with no search.

In the above, the y-axis is a measure of skill. A skill-level of 13 means that the agent has an approximately 50% win-rate
against a 13-ply exhaustive tree-search agent. Given that each player makes a maximum of 21 moves in Connect4, 21-ply
exhaustive tree-search represents perfect-play, meaning that the dashed line at y=21 represents perfect play. The above
plot thus indicates that the system attains optimal results against perfect play within 5 hours (i.e., it always wins as
first player against perfect play).

You can also manually play against an MCTS agent powered by a net produced by the AlphaZero loop. For the above Connect4
example, you can do this with a command like:

```
./target/Release/bin/c4 --player "--type=TUI" \
  --player "--type=MCTS-C -m /workspace/output/c4/my-first-run/models/gen-10.pt"
```

## C++ Overview

### Directory Structure

In the `cpp/` directory, there are 3 high-level subdirectories:

```
cpp/include/
cpp/inline/
cpp/src/
```

The `include/` directory contains the header files (`.hpp`), and the `src/` directory contains independently compiled
`.cpp` files. For headers that contain template functions, the implementations of those functions typically live in an
inline file (`.inl`) - those files live in the `inline/` directory, and are `#include`'d at the bottom of the
corresponding header file. All three directories have parallel matching subdirectory structures.

Below are the list of modules of `cpp/include/`. In the list, no module has any dependencies on a module that appears
later in the list.

* `third_party`: third-party code that was simply copy-pasted because it was not available as a package
* `util`: utility code that is not specific to games/AlphaZero
* `core`: core game code (nothing MCTS-specific). Some key classes provided here:
  * `AbstractPlayer`: abstract class for a player that can play a game
  * `GameServer`: runs a series of games between players (which can optionally join from other processes)
* `mcts`: generic MCTS implementation
* `games`: game-specific types and players. Each game (e.g., connect4, othello) has its own subdirectory
* `generic_players`: generic player implementations that can be used for any game

### Game Types as C++ Template Parameters

The MCTS code is entirely templated based on the game type. This can make the code a bit daunting at first. What drove
this decision?

A high-performance MCTS implementation should aim to saturate both GPU and CPU resources via parallelism. When CPU
resources are fully saturated, it is common for the PUCT calculation that powers MCTS to become a bottleneck. In order
to optimize this calculation, it is important for the various tensors involved to have sizes and types known at compile
time. If the sizes and types are specified at runtime, then the tensor calculations can hide a lot of inefficient
dynamic memory allocation/deallocation and virtual dispatch under the hood.

Fundamentally, this consideration drove the design of this framework to specify the game type as a template parameter.
The simpler alternative would have been to use an abstract game-type base class and inheritance, but this would incur
the performance penalty described above.

Note: most MCTS implementations are for 1-player games or 2-player zero-sum games. In such games, the value of a state
can be represented as a scalar. This implementation, however, supports n-player games for arbitrary n, and so the value
is instead represented as a 1D tensor. This is another reason why compile-time knowledge of the game type helps,
as otherwise, all value-calculations (which are simply scalar calculations in typical MCTS implementations) would incur
dynamic memory allocation/deallocation.
