# Scrabble

This document sketches out a plan for tackling the game of Scrabble.

## Background

Our goal is to create the first superhuman (by 2024 standards) Scrabble AI.

Quackle is an open source program considered to be one of the strongest in the world. During the midgame, it works as follows:

- From any given board state and rack, every candidate move is enumerated, and that move’s _equity_ is computed.
Equity is $(A + B)$, where $A$ is the move's score value, and where $B$ is an estimate of the potential future-value of the
move's _leave_ (Scrabble parlance for the tiles remaining on the rack after the move is played). 
- For the top $k$ equity moves, a Monte Carlo simulation is performed, traversing a shallow depth down the game tree.
Each traversal samples the opponent's rack uniformly randomly from the unseen tiles, and picks the top-equity move
for each player. Of the $k$ tested moves, the one that performs best is chosen, measured by score-differential + equity-differential.

The leave-estimator is either simple linear regression model taking subsets of the leave as input, or
a full lookup table for every possible leave. It is generated by some sort of offline simulation process.
This table reflects the fact that certain letter combinations provide stronger bingo possibilities (e.g., -ING, -EST), that
a healthy ratio of vowels-vs-consonants is preferable, that repeated letters reduce the number of combinatorial rack possibilities,
that a Q is better when accompanied by a U, etc.

Grandmaster Kenji Matsumoto has detailed the weaknesses of this approach on this [page](http://www.breakingthegame.net/computers5). Summarizing his observations:

- The AI has poor long-term planning, not willing to hold strong tiles for multiple turns.
- The uniform random assumption of the opponent’s rack is naive. In reality an opponent’s play gives us important information about their likely remaining tiles.
- The leave-estimator is ignorant of board dynamics.
- Equity maximization fails to navigate the expectation-vs-variance tradeoff.

This excellent [video](https://youtu.be/oBmnpNwqE48?si=LG_PQzKs3VDRP1TW&t=276) by world-class Scrabble player Will Anderson
illustrates a situation where Quackle performs poorly, discussing in detail the AI mechanics described above. The
relevant section starts at 4:36 (link starts there) and ends at 7:04.

A principled, game-theoretically-sound approach, that converges towards the game’s Nash Equilibrium, should naturally overcome all these shortcomings.

In the endgame, Scrabble is a game of perfect information, like chess. As such, Quackle switches to a different mode, both
for the endgame, and for the pre-endgame. In theory, perfect-information routines like Alpha-beta pruning should work for the
endgame, but in practice, the game tree size is too large, and crafting a heuristic evaluation function to estimate leaf nodes of a partial
subtree is difficult. So Quackle uses B* search to sparsely explore the game tree. See this excellent blog 
[post](https://medium.com/@14domino/scrabble-is-nowhere-close-to-a-solved-game-6628ec9f5ab0) by Scrabble expert César Del Solar
for a technical discussion of empirical shortcomings of this approach.

For our planned AlphaZero-based implementation, the hope is that a single unified
algorithm can work across all phases of the game, with no need for endgame specialization. After all, "the game tree size is too large" and "crafting a heuristic evaluation
function is difficult" are exactly the reasons that Alpha-beta pruning failed at go, and AlphaZero was the answer there. If
this hope turns out be misplaced, we will later explore endgame specializations.

## Q-Dispersion Tree Search (QDTS)

We have devised a variant of MCTS that, when used in an AlphaZero loop, is guaranteed to yield
approximate policy convergence towards Nash Equilibrium. We have named this variant _QDTS_. 
See [here](Q-Dispersion-Tree-Search.md) for a full description.

In QDTS, we require a hidden-state-network, $H$, that samples the hidden state of the game.
In Scrabble, you can consider the opponent's entire rack as the hidden state, or you can consider just the leave. We choose to
use leaves rather than entire racks, as the training targets will be sharper without the diluting
effect of the uniform random bag replenishment.

There are $>1e7$ possible leaves in Scrabble, and outputting a logit distribution over all of them is likely unwieldy. 
Instead, we can have $H$ sample the leave $t$ tiles at a time, which only requires an output logit layer of size $27^t$.
We can sample the entire $T$-tile leave by performing $\lceil T/t \rceil$ queries to $H$, including the partially produced rack
as part of the input.

## Details

### Action Representation

The number of possible moves in Scrabble can be large. One natural candidate move representation is of size $15 * 15 * 2 * 7! > 2e7$.
The move location is encoded by the $(15, 15)$, the direction (horizontal or vertical) by the $2$, and the ordering of tiles
by the $7! = 840$. This omits the letter-instantiation of the blank tiles, which technically can add another factor of $26^2$.

Besides this number being too large to reasonably use for an output logit layer, asking the network to learn to encode
size-7 permutations is a tall task.

We can instead adopt an approach that has been proven to work in other high-branching-factor games like Arimaa: decompose the
move into _submoves_, and reformulate the game rules so that the current player takes multiple turns in a row. In Scrabble,
we can for example decompose a real move into multiple submoves:

1. Choose a board location + direction $(15, 15, 2)$
2. Choose tile 1 $(53)$
3. Choose tile 2 $(53)$
4. ...

The 53 here represents the 26 standard tiles and the 26 possible ways to play a blank tile, plus an extra 1 to indicate
a null-terminator. This reduces the output layer size to a much more manageable size, while also eliminating the need for the network to
learn permutation encodings.

(We omit pass/exchange moves here for simplicity of exposition.)

However, this decomposition has an undesirable property. To illustrate, consider holding a rack of `AEFINTS` in the starting
position of the game. The bingo `8D FAINEST` is the best play. But this shares the same tree-prefix as the move `8D FATES`
(specifically, they share the location/direction of `8D` and the first two tiles of `FA`). So the MCTS backpropagation step
for the move `8D FAINEST` ends up incentivizing the move `8D FATES`, when the two moves don't have anything to do with each
other.

More broadly, a good move decomposition scheme ideally has the property that if a move is good, then other moves that
share a decomposed-prefix are also likely to be good.

We can improve the decomposition scheme by instead making the number of tiles in the move part of submove 1:

1. Choose a board location + direction **+ move-size** $(15, 15, 2, 7)$
2. Choose tile 1 $(52)$
3. Choose tile 2 $(52)$
4. ...

In this scheme, `8D FAINEST` and `8D FATES` no longer share a decomposed-prefix.

There may be other decomposition schemes that are better than this one. For example, we can first choose the subset of
letters to be used (there are $2^7 = 128$ possibilities), followed by the location, followed by the individual tiles.
It could be better to join the subset and the location, for a size of $(128, 15, 15)$. Experimentation is needed.

### Network Requerying Optimization

Repeatedly querying the network to generate a move in chunks, or to sample the hidden information in chunks, makes MCTS substantially
costlier. It may be possible to mitigate this by decomposing the network into two subnetworks.
The first, costlier subnetwork produces some compact internal representation.
The second, cheaper subnetwork, accepts this internal representation as input, along with the partially produced
output, and produces the next part of the output. Rather than querying a single expensive network $k$ times, then, we
query an expensive subnetwork once, and a cheap subnetwork $k$ times.

The Epinet ([Osband et al, 2023](https://arxiv.org/pdf/2107.08924.pdf)) has an architecture like this, although
their motivation is different from ours.

### Legal Move Mask

In Scrabble, it may be difficult for the network to learn the set of legal moves in a given position. Doing this
perfectly demands memorizing the entire lexicon, and it is unlikely the network can learn this through self-play,
given that many words are unlikely to come up even once over the course of, say, 1 billion self-play games.

We can consider helping the network along by first computing a mask of legal (sub-)moves, and then passing
that mask as part of the input. The network should quickly learn that it should only place output mass on
(sub-)moves where the mask is 1, and that moves where the mask is 1 should be considered when evaluating the
value of the current position.

The [GADDAG](https://en.wikipedia.org/wiki/GADDAG) data structure supports efficient calculation of a mask of all legal
moves given a board and a rack. We need to compute GADDAG masks anyways to mask network policy outputs,
so passing the masks as a network input should not represent a significant extra cost.

### Leaf Rollouts

In standard AlphaZero MCTS, when you reach a leaf node $n$, you obtain a $V$ estimate from the network, and backpropagate
that estimate up the tree. We can instead do a shallow rollout from $n$, down to some descendant $d$, generate
our estimate at $d$, and then backpropagate that estimate from $n$ to the tree root. The descent
from $n$ to $d$ can be done by sampling from $P$ (and $H$ and the bag) at each step. This generalizes Quackle's
Monte Carlo sampling mechanics.

Without these leaf rollouts, we may be overly sensitive to inaccuracies in $V$. In general, for $V$ to be perfect,
the network requires a representation of the entire lexicon, which runs into the problems mentioned above. If
the current/projected game state involves rare words that the network's implicit internal lexicon does not yet know,
$V$ could be deficient. Rollouts help us smoothen out this problem, since the moves along the rollout will be appropriately
informed by GADDAG-calculated move masks.

A downside of these leaf rollouts is that the randomness of the $H$-sampling and the bag-replenishing from $n$ to $d$
can introduce noise to the leaf evaluation. This is partially mitigated by the variance-reduction that comes
from our alternative $Q$ formulation, combined with our action-value prediction mechanism (TODO: describe these).
