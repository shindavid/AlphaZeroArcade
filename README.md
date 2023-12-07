
# Alpha Zero Arcade

A generic AlphaZero framework.

There are many AlphaZero implementations out there, but most of them are for a specific game. Some implementations are 
more generic but at a serious performance cost. This implementation is designed to be maximally generic at minimal
overhead.

The implementation aims to incorporate as many state-of-the-art ideas and techniques as possible from other projects.
In particular, it borrows heavily from [KataGo](https://github.com/lightvector/KataGo). Eventually, it hopes to work
just-as-well as KataGo does for go, while minimizing go-specific details in its implementation.

## Getting Started

### Initial setup

The project assumes you are working on a Linux platform. No other OS's will be supported.

It is recommended to create a conda environment for this repo, with a command like:

```
$ conda env create -n env_name -f environment.yml
```

After creating and activating the conda environment, you must run the following command from the repo root one-time to
get your local python imports to work properly:

```
$ cd py; python setup.py develop; cd -
```

There are then a few third-party dependencies that are not available in conda/pip that you must download manually. Currently,
these are libtorch, EigenRand, and tinyexpr. The script `extra_deps/update.sh` downloads those packages and installs them
in the `extra_deps/` directory. You can install them  into `extra_deps/` by running the script or by manually running a
subset of the install commands from the script as needed. Alternatively you can install them elsewhere, or point to existing
locations if you already have one or more of these packages installed on your machine.

If you wish to test against third-party reference agents for Connect4 or Othello, you need to download other packages
and build those separately as well; more on that later.

A few notes:

[1] Why is a separate libtorch download needed, you may ask? The version of libtorch that you get from conda/pip unfortunately is
of the Pre-cx11 ABI variety, while this project requires the cx11 ABI variety. Note that the script has CUDA 11.8 hard-coded;
depending on your hardware you might need a different version. The download page is [here](https://pytorch.org/get-started/locally/).

[2] TODO: for the other dependencies, I should fork them and then have `update.sh` fetch from the fork, in order to have
full control over the exact version. Perhaps using git subtree or similar would be appropriate.

Finally, you should create a `config.txt` file at the root of your repo checkout with at least the following:

```
libtorch_dir = /path/to/libtorch
eigenrand_dir = /path/to/EigenRand
tinyexpr_dir = /path/to/tinyexpr

alphazero_dir = /path/where/you/want/to/write/data/files
```

### Building

From the repo root, run:

```
./py/build.py
```

Pass `-j 8` to `build.py` to use 8-fold parallelism (alternatively, add `cmake.j = 8` to your `config.txt`).

This should build a binary for each supported game, along with some unit-test binaries. The end of the output should list
the available binary paths:

```
...
Binary location: target/Release/bin/c4
Binary location: target/Release/bin/othello
Binary location: target/Release/bin/othello_unit_tests
Binary location: target/Release/bin/tictactoe
Binary location: target/Release/bin/util_unit_tests
```
You can then run for example `target/Release/bin/tictactoe -h` to get a list of help options.

### Running the AlphaZero loop

After building the binary, you can launch the AlphaZero loop for the game of your choice with a command like this:

```
./py/alphazero/main_loop.py --synchronous-mode --game tictactoe --tag my-first-run
```
or using aliases,
```
./py/alphazero/main_loop.py -S -g tictactoe -t my-first-run
```

Here `my-first-run` is a run _tag_. All files produced by the run will then be placed in the directory 

```
${alphazero_dir}/tictactoe/my-first-run/
```

where `${alphazero_dir}` is the path you specified in your `config.txt`.

The `--synchronous-mode` option makes it so that the script alternates between running a fixed number of self-play games
and then performing a fixed number of neural net train updates. The default behavior is asynchronous, meaning that the
self-play and the neural-net training happen simultaneously. The default behavior requires you to have multiple CUDA
devices on your machine (one is dedicated to self-play and one is dedicated to neural-net training).

### Measuring Progress

You can manually play against an MCTS agent powered by a net produced by the AlphaZero loop. For the above tictactoe
example, this can be done by something like:

```
./target/Release/bin/tictactoe --player "--type=TUI" --player "--type=MCTS-C -m ${alphazero_dir}/tictactoe/my-first-run/models/gen-10.ptj"
```

For something more systematic, you want to run the MCTS agents against a family of reference agents. For this, you can run:

```
./py/alphazero/compute_ratings.py -D -g tictactoe -t my-first-run
```

This launches a dameon that continuously run matches between the produced family of MCTS agents against a family of
reference agents, storing match results and estimated ratings into an sqlite3 database. This process too uses a GPU,
so if you don't have a free GPU on your machine, it is recommended to run it while the main loop is not running, or
to run it on a separate machine (your `${alphazero_dir}` should then be on a network-mounted filesystem).

To then visualize the data, you can run:

```
./py/alphazero/viz_ratings.py -g tictactoe -t my-first-run
```

This will launch an interactive bokeh plot in your web-browser. A sample plot can be seen below.

### Reference Agents for Connect4 and Othello

If doing the above instructions for c4 or othello instead of tictactoe, you need to download and build binaries.
The instructions for this can be found in 
[`py/othello/README.md`](https://github.com/shindavid/AlphaZeroArcade/blob/main/py/othello/README.md) and 
[`py/connect4/README.md`](https://github.com/shindavid/AlphaZeroArcade/blob/main/py/connect4/README.md).

## Example plots

Here is a plot showing learning progress for the game of Connect4:

![image](https://github.com/shindavid/AlphaZeroArcade/assets/5217927/a8c1edb8-425e-4634-803f-086801aa59cd)

The dark curve corresponds to an MCTS agent using i=1600 iterations per search. The light curve corresponds to an agent
that plays according to the raw network policy with no search.

In the above, the y-axis is a measure of skill. A skill-level of 13 means that the agent has an approximately 50% win-rate
against a 13-ply exhaustive tree-search agent. Given that each player makes a maximum of 21 moves in Connect4, 21-ply
exhaustive tree-search represents perfect-play, meaning that the dashed line at y=21 represents perfect play. The above
plot thus indicates that the system attains optimal results against perfect play within 5 hours (i.e., it always wins as
first player against perfect play).

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
* `games`: game-specific types and players. Each game (e.g., connect4, othello) has its own subdirectory. There is a
  `generic/` subdirectory which contains generic player implementations that can be used by other games.

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
