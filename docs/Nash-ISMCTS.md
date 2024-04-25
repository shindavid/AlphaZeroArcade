# Nash Information Set Monte Carlo Tree Search (Nash-ISMCTS)

## Background: Many Tree ISMCTS (MT-ISMCTS)

In perfect information settings, each node of an MCTS tree represents a full game state. In imperfect
information settings, the acting agent lacks visibility into all hidden information, and so cannot
construct such a tree.

[Blüml et al, 2023](https://www.frontiersin.org/articles/10.3389/frai.2023.1014561/full)
provides a comprehensive survey of various workarounds. Broadly, there are two approaches:

1. _Determinize_ the game by converting the imperfect information game into a perfect information one,
learn a policy for this game using perfect-information tree-search techniques, and then somehow
convert the perfect-information strategy into an imperfect-information one.

2. Construct an _information set_ MCTS (ISMCTS) tree, where each node represents an _information set_. This is the
part of the game state that is visible to the acting player. Devise tree search mechanics that operate
on this tree.

Of the second category of approaches, there is Many Tree Information Set MCTS (MT-ISMCTS),
introduced by [Cowling et al, 2015](http://orangehelicopter.com/academic/papers/cig15.pdf).
This will serve as the starting point of our planned implementation. 

Here is an illustration of MT-ISMCTS mechanics:

![image](https://github.com/shindavid/AlphaZeroArcade/assets/5217927/b3dca415-e51c-485f-b7f0-d8871f6b8940)

The above diagram depicts a game with two players. The first player is Alice, and the second player is Bob.
Red nodes correspond to information sets where it is Alice's turn, and blue nodes correspond to information
sets where it is Bob's turn. Each node contains two symbols, representing Alice's private information
and Bob's private information, respectively.

Let us start at the top left red node labeled `x?`. It is Alice's turn. The node is contained in a red
tree, which indicates that the calculations are from Alice's point-of-view (POV). She performs her action
according to an action selection criterion (ACT). Here, she selects the right action, along the edge
labeled `N_b`.

Alice now wishes to simulate Bob's action, as would typically be the next step in standard MCTS. However,
without Bob's private information, Alice cannot simulate his action. Thus, Alice _samples_ Bob's
private information, according to some belief distribution. Here, she samples that his hidden
information is `z`.

She is now able to simulate Bob's action. However, she must account for the fact that Bob does not know
Alice's private information. She thus _spawns_ a new tree, from Bob's POV. This tree is colored blue, to
indicate that it corresponds to Bob's POV. The root of this node is labeled `?z`, to reflect the fact that
Bob knows his private information and does not know Alice's information. The simulation of Bob's action
recursively repeats the same procedure that we started with.

Bob's recursive application of this procedure can itself spawn a new tree from the already-spawned tree,
represented by the third tree in the diagram. And this recursive spawning can continue indefinitely.

This repeated spawning is important. In Alice's simulation of Bob's thinking, she must conceive of him
as an agent that believes Alice can have non-`x` hidden-states, optimizing his action accordingly. The
`Q` values relevant to her decision-making should reflect her own known hidden information of `x`, but
the `Q` values relevant to Bob's decision-making should not. Furthermore, her simulation of Bob's thinking
is an agent that is simulating her own thinking. The simulated-Alice within Alice's simulation of Bob
must potentially have non-`x` states. And so forth.

As [Cowling et al, 2015](http://orangehelicopter.com/academic/papers/cig15.pdf) predated AlphaGo/AlphaZero,
some of the details of the MCTS mechanics were different from what they would be in a more modern
implementation:

- Leaf values were obtained via random rollout rather than via a neural network evaluation
- The action-selection criterion did not incorporate a policy prior, and used a different formula
from PUCT.

## Hidden State Model

We will now build up towards our modified version of MT-ISMCTS, which we will call _Nash-ISMCTS_.
We start with the obvious AlphaZero-based modernizations:

- Replace random rollouts with value network (`V`) evaluations
- Incorporate a policy network (`P`) and use PUCT as the action selection criterion.

An important detail omitted in our description of MT-ISMCTS is the SAMPLE step. How should Alice sample Bob's hidden information?
[Cowling et al, 2015](http://orangehelicopter.com/academic/papers/cig15.pdf)
propose a variety of approaches to sample this information, with accompanying experimental results.
Instead of adopting one of their proposed approaches, we will instead use an
AlphaZero-inspired approach: train a _hidden-state_ neural network `H`
that learns to sample the hidden state of the game. Note that in principle, `H` can be computed exactly from `P` via
Bayes' Rule, but this computation can be expensive. So `H` can be considered an alternate representation of `P` that we
use to avoid an expensive online Bayes' Rule calculation.

`H` will accept the same inputs as `P` and `V`, and will be trained on self-play games using the actual values of the hidden state
as training targets. This should result in an unbiased `H` as long as the self-play games are played according to
the policy `P`. We can approximately enforce this by being careful with move temperature mechanics; more on this later.

In some games, the size of the hidden state space can be intractably large. In Scrabble, for instance, there are about 3e7
possible racks. Rather than modeling the policy as a single logit distribution over entire space, we can have `H`
generate samples in chunks, similar to how an LLM samples sentences one token at a time. In Scrabble, we can
generate a hidden rack one letter at a time, limiting the output to a size-27 logit layer.

## Equilibrium-Idempotence

If we play self-play games using Nash-ISMCTS with `n` visits, and train `P`, `V` and `H` on the resultant set of 
self-play data, can we expect convergence to Nash Equilibrium, as `n` and the number of self-play games approache infinity?

Unfortunately, the answer is likely no, as the resultant dynamic system will exhibit unstable chaotic behavior. However,
in analyzing some of the properties of this dynamic system, we can obtain some useful insights that will help us
design a more sound version.

We start with a theoretically interesting property of the system: if `P`, `V`, and `H` exactly match Nash Equilibrium,
then the visit distribution `N` produced by Nash-ISMCTS will approach `P` as `n` approaches infinity. When viewing
MCTS as an operator mapping policies to policies, we can say that Nash-ISMCTS is idempotent at equilibrium, or that
it exhibits _equilibrium-idempotence_.

To see this, note that the PUCT formula is:

```
PUCT(a) = Q(a) + c * P(a) * sqrt(sum(N)) / (1 + N(a))
```

By standard properties of Nash equilibria, the assumption of equilibrium translates to the following: 
`P(a) == 0` for all `a` not in `S`, where `S` is the set of actions `a` for which `V(a)` attains it maximum
of `v_max`.

If `V` is accurate, as presumed by the equilibrium hypothesis, then `Q(a)` at actions in `S` should converge
towards `v_max`, while `Q(a)` at actions not in `S` should converge towards values less than `v_max`. Since
`Q` dominates the equation as `N(a)` approach infinity, the proportion of `N` on actions not in `S`
should approach 0. It remains to show that the ratio `N(a_i) / N(a_j)` approaches `P(a_i) / P(a_j)`
for all `a_i, a_j` in `S`.

To see this, we can plug `a_i` and `a_j` into the `PUCT` equation, and set PUCT values equal. The `Q(a)`, 
`c`, and `sqrt(sum(N))` terms all cancel, leaving us with:

```
P(a_i) / P(a_j) = (1 + N(a_i)) / (1 + N(a_j))
```

This demonstrates the required ratio limit, and thus proves the claimed property of equilibrium-idempotence.

## Practical Convergence to Equilibrium

The theoretical argument above demonstrates that if we are at equilibrium, then MT-ISMCTS will
converge towards an idempotent operator. However, we must be _exactly_ at equilibrium. If `V` or `H` are off by
even the tiniest of margins, then the idempotence proof fails, and as the number of MCTS iterations approaches
infinity, the visit distribution will collapse to a single point, rather than converge to `P` as required.

Practically, the number of visits is finite, and so if the networks are close enough, then perhaps the visit
distribution will be close enough to `P` to make everything work in practice. However, this is not satisfactory. One should
have confidence that the quality of the posterior policy will increase as a function of the number of visits. Without
such a guarantee, it is difficult to trust that AlphaZero will result in long-term improvement.

We have brainstormed many potential remedies for this issue. Currently, our most promising remedy is this:

- For hidden-state nodes, compute `Q` as a weighted-sum of the children `Q`, and use `H` for the weights of that
average. In the above diagram, this means that `Q` at node `b` is the weighted sum of the `Q` values of nodes `c`
and `d`, with `H` dictating the weights of that sum.

- Consider the _eps-neighborhood_, `N_eps(H)`, of `H`, comprising all other hidden-state-distributions whose distance from `H`
is bounded by some small constant `eps`. Formally, if `H` is a probability distribution over `k` items, it can be
represented as a point in the `k`-simplex. Then, `N_eps(H)` can be the set of points of the `k`-simplex whose L1-distance
to `H` is at most `eps`.

- When performing PUCT at action nodes, compute the selection criterion for hidden-state children nodes by using
every single `H'` in `N_eps(H)`, rather than using just a single `H`. This yields a set of possible actions, rather than
just a single one.

- If this set is of size one, then just select that one action. Otherwise, among the actions in the set, select one
at random, selecting `i` with probability proportional to `P(i)`.

It is not difficult to show that this criterion guarantees equilibrium-idempotence, even after applying small perturbations
to `H` or `V`, thus resolving the problematic instability.

## Experiments

TODO: Describe Kuhn poker experiments validating the theory described here.
