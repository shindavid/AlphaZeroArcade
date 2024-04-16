
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

### Env setup

The project assumes you are working on a Linux platform. No other OS's will be supported.

To get started, clone the repo, and then run:

```
$ source env_setup.sh
```

This will launch `setup_wizard.py`, which will walk you through the necessary installation steps. These steps may
require some manual external steps like installing CUDA and torchlib. Such steps unfortunately cannot be automated
because of licensing reasons.

In the future, whenever you open a new shell, you should rerun the above command. Subsequent runs will be much faster,
as they simply define some environment variables and activate a conda environment.

### Building

From the repo root, run:

```
./py/build.py
```

This should build a binary for each supported game, along with some unit-test binaries. The end of the output should list
the available binary paths:

```
...
Binary location: target/Release/bin/c4
Binary location: target/Release/bin/othello
Binary location: target/Release/bin/othello-tests
Binary location: target/Release/bin/tictactoe
```
You can then run for example `target/Release/bin/tictactoe -h` to get a list of help options.

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
./py/alphazero/scripts/run_local.py --game tictactoe --tag my-first-run
```
or using aliases,
```
./py/alphazero/scripts/run_local.py -g tictactoe -t my-first-run
```
This launches one instance of each the 3 server types (loop-controller, self-play, ratings).

Here `my-first-run` is a run _tag_. All files produced by the run will then be placed in the directory 

```
$A0A_OUTPUT_DIR/tictactoe/my-first-run/
```

where `$A0A_OUTPUT_DIR` is an environment variable configured during env setup.

By default, `run_local.py` will detect the number of available cuda devices on the local machine, and allocate
the GPU's across the servers in an optimal manner. If 1 or more GPU's are shared by multiple servers, the
loop-controller carefully orchestrates the entire process, pausing and unpausing components as needed to ensure
that the GPU's stay fully utilized, without the components thrashing with each other or getting starved indefinitely.

### Measuring Progress

During-or-after a run of the loop-controller, you can launch a web dashboard to track the progress of your run:

```
./py/alphazero/scripts/launch_dashboard.py -g tictactoe -t my-first-run --open-in-browser
```

This launches an interactive dashboard in your web browser, which currently looks like this:

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

You can also manually play against an MCTS agent powered by a net produced by the AlphaZero loop. For the above tictactoe
example, you can do this with a command like:

```
./target/Release/bin/tictactoe --player "--type=TUI" \
  --player "--type=MCTS-C -m $A0A_OUTPUT_DIR/tictactoe/my-first-run/models/gen-10.pt"
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

## Just

A command-runner called `just` is used for various scripting.

Install instructions are here: https://github.com/casey/just#packages

To view available commands, just run: `just`

## Docker and Cloud GPUs

Below are some instructions for building and running on GPUs in the cloud.

(Note: this was only tested on the lambdalabs cloud, but is easily extensible to others)

Steps:
  1. Create instance and setup ssh config for HOSTNAME
  2. Run `just setup-lambda HOSTNAME` to configure node, build docker container, install all deps (takes about 5-10 minutes)
  3. Run `just goto HOSTNAME` to log into the cloud docker container
  4. Run `just build` to build
  5. Run `just train_c4 YOUR_TAG_HERE -S` to train connect-4
