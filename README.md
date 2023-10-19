
# Alpha Zero Arcade

A generic AlphaZero framework.

There are many AlphaZero implementations out there, but most of them are for a specific game. Some implementations are 
more generic but at a serious performance cost. This implementation is designed to be maximally generic at minimal
overhead.

## C++ Overview

### Directory Structure

In the below list of modules in the `cpp/` directory, no module has any dependencies on a module that appears later in
the list.

* `third_party`: third-party code that was simply copy-pasted because it was not available as a package
* `util`: utility code that is not specific to AlphaZero
* `core`: generic code for games (nothing MCTS-specific). Some key classes provided here:
  * `AbstractPlayer`: abstract class for a player that can play a game
  * `GameServer`: runs a series of games between players (which can optionally join from other processes)
* `mcts`: generic MCTS implementation
* `common`: provides `AbstractPlayer` derived classes that work for any game type
* `games`: game-specific types and players. Each game (e.g., connect4, othello) has its own subdirectory

### Game Types as C++ Template Parameters

The MCTS code is entirely templated based on the game type. This can make the code a bit daunting at first. What drove 
this decision?

A high-performance MCTS implementation should aim to saturate both GPU and CPU resources via parallelism. When CPU
resources are fully saturated, it is common for the PUCT calculation that powers MCTS to become a bottleneck. In order
to optimize this calculation, it is important for the various tensors involved to have sizes and types known at compile
time. If the sizes and types are specified at runtime, then the tensor calculations can hide a lot of inefficient
dynamic memory allocation/deallocation under the hood.

Fundamentally, this consideration drove the design of this framework to specify the game type as a template parameter.
The simpler alternative would have been to use an abstract game-type base class and inheritance, but this would incur
the performance penalty described above.

Note: most MCTS implementations are for 1-player games or 2-player zero-sum games. In such games, the value of a state
can be represented as a scalar. This implementation, however, supports n-player games for arbitrary n, and so the value
is instead represented as a 1D tensor. This is another reason why compile-time knowledge of the game type is important,
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

## Example plots

Here is a plot showing learning progress for the game of Connect4:

![image](https://github.com/shindavid/AlphaZeroArcade/assets/5217927/58b52859-75ed-4830-950f-893b5473f3d3)

The agent being tested is an MCTS agent using i=300 iterations per search.

In the above, the y-axis is a measure of skill. A skill-level of 13 means that the agent has an approximately 50% win-rate
against a 13-ply exhaustive tree-search agent. Given that each player makes a maximum of 21 moves in Connect4, 21-ply
exhaustive tree-search represents perfect-play. Given furthermore that the first player is provably winning in Connect4
with perfect play, the dashed line at y=21 represents perfect play. The above plot thus indicates that the system learns
to master Connect4 at 300 mcts iterations in about ~5 hours. The slight dips in the curve after hitting y=21 can be
attributed to a minor injection of randomness in the agent's first R moves, and disappears if we increase from i=300
to i=1600 or more.
