# Information Set Monte Carlo Tree Search

## Background

In perfect information settings, each node of an MCTS tree represents a full game state. In imperfect
information settings, the acting agent lacks visibility into all hidden information, and so cannot
construct such a tree.

[Blüml et al, 2023](https://www.frontiersin.org/articles/10.3389/frai.2023.1014561/full)
provides a comprehensive survey of various workarounds. Broadly, there are two approaches:

1. _Determinize_ the game by converting the imperfect information game into a perfect information one,
learn a policy for this game using perfect-information tree-search techniques, and then somehow
convert the perfect-information strategy into an imperfect-information one.

3. Construct an _information set_ MCTS (ISMCTS) tree, where each node represents an _information set_. This is the
part of the game state that is visible to the acting player. Devise tree search mechanics that operate
on this tree.

Of the second category of approaches, there is Many Tree Information Set MCTS (MT-ISMCTS),
introduced by [Cowling et al, 2015](http://orangehelicopter.com/academic/papers/cig15.pdf).
This will serve as the starting point of our planned implementation. 

Here is an illustration of MT-ISMCTS mechanics:

![ISMCTS2](https://github.com/shindavid/AlphaZeroArcade/assets/5217927/31141cd9-431f-443d-88b5-7480cb1203ba)

In the above, we have a tree, `T`, with root node `a`. There are two players in this game, `P1`
and `P2`. Each node of the tree contains two symbols, representing `P1`'s private information, followed by
`P2`'s private information.

At `a`, it is `P1`' action. `P1` has private information `x`, and `P2`'s private information is unknown.
`P1` acts at `a` via PUCT, to arrive at node `b`. Here, we want to
simulate `P2`'s action, but that action is dependent on `P2`'s private information, which we
do not have. Thus, we sample `P2`'s private information according to some hidden-state policy `H`.
This sampling results in node `d`, where `P2`'s hidden information is instantiated to `z`.

At `d`, to further descend down the tree, we need to model `P2`'s decision. To do this, we
spawn a new tree, `U`, and recursively perform the same mechanics. At `f`, the root of this spawned tree,
we mask `P1`'s private information to model `P2`'s POV. We perform PUCT as we did at `a`, leading to
node `g`.

Back in the original tree `T`, we obtain the PUCT-selected-action at `f`, and use that action
to continue descending to node `e`.

Repeating this entire procedure leads to visit distributions at the children of `b`. Those distributions
can be combined to yield a visit distribution at `b` itself, which yields combined children of `b` of the
form `x?`, at which the same overall routine performed at `a` can be repeated, to go arbitrarily
deep into the game-tree within `T`.

It is important to note that the `Q` values at `e` and `g` will be different. The `e` values reflect
the known information of `P1`, while the `g` values do not. The `Q` values obtained in `U` are
irrelevant to `T`; only the action-selections are of interest.

This description of MT-ISMCTS is slightly anachronistic, as [Cowling et al, 2015](http://orangehelicopter.com/academic/papers/cig15.pdf)
predates AlphaGo/AlphaZero. PUCT was not the favored selection criterion at the time, and leaf
evaluations were performed by random rollouts, rather than via neural network evaluations. The
important part here is the mechanics of tree-spawning.

## Hidden State Model

An important detail omitted in the above is the hidden-state-sampling policy `H`. [Cowling et al, 2015](http://orangehelicopter.com/academic/papers/cig15.pdf)
propose a variety of approaches to sample this information, with accompanying experimental results.
Instead of adopting one of their proposed approaches, we will instead use an
AlphaZero-inspired approach: train a _hidden-state_ neural network
that learns to sample the hidden state of the game. Note that in principle, `H` can be computed exactly from `P` via
Bayes' Rule, but this computation can be expensive. So `H` can be considered an alternate representation of `P` that we
use to avoid an expensive online Bayes' Rule calculation.

`H` will accept the same inputs as `P` and `V`, and will be trained on self-play games using the actual values of the hidden state
as training targets.

In some games, the size of the hidden state space can be intractably large. In Scrabble, for instance, there are about 3e7
possible racks. Rather than modeling the policy as a single logit distribution over entire space, we can have `H`
generate samples in chunks, similar to how an LLM samples sentences one token at a time. In Scrabble, we can
generate a hidden rack one letter at a time, limiting the output to a size-27 logit layer.

## Theoretical Convergence to Equilibrium

If we play self-play games using MT-ISMCTS with `n` visits, and train `P`, `V` and `H` on the complete resultant set of 
self-play data, can we expect convergence to Nash Equilibrium, as `n` approaches infinity?

Formally, if we imagine the combined weights of the `P`, `V`, and `H` networks to be a point in `R^d`, then the generations
of training yields a path-like sequence of points in `R^d`: `x_1, x_2, ...`. There is some subset `NE` of `R^d`
representing the game's Nash equilibria. Does `x_i` have a limit, and if so, is that limit in `NE`?

Here is a soft-proof that the answer is yes.

The proof entails 3 parts:

1. **(Equilibrium-Idempotence)**: If currently at equilibrium, we will stay there.
2. **(Non-Complacency)**: If not currently at equilibrium, we cannot stay there.
3. **(Limit-Existence)**: The limit of `x_i` must exist.

These 3 assertions clearly prove our desired result. Let us prove them in turn.

#### Equilibrium-Idempotence

Suppose that `P`, `V`, and `H` are at equilibrium.

The PUCT formula is:

```
PUCT(a) = Q(a) + c * P(a) * sqrt(sum(N)) / (1 + N(a))
```

The idempotence requirement means that as the number of iterations approaches infinity, the distribution of `N`
should approach the same shape as `P`. 

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

This is enough. We can make the convergence better by replacing the `(1 + N(a))` term in the PUCT formula with
`max(c, N(a))` for some fixed constant `0 < c < 1`.

Some practicalities should be noted. Root softmax temperature must be disabled for this proof to work.
Dirichlet noise also breaks the analysis. Finally, move-temperature has the potential to make the agent
act with non-equilibrium frequencies. It is unclear if this last point will cause theoretical problems during
self-play, but certainly during competitive play, failure to act with equilibrium frequencies will lead to an
exploitable agent. Later, we will investigate how to deal with these issues practically.

#### Non-Complacency

Suppose `P`, `V` and `H` are not at equilibrium. Then, there are game states at which they produce estimates that
are not at equilibrium with respect to their estimates at their descendants. Of these misestimated states,
choose a node `n` that is closest to a terminal game state.

The networks cannot converge to this state, since `MT-ISMCTS` will produce a policy that exploits the
misestimates at `n`, and this exploitation corresponds to a difference between `P` and `N` that will cause network drift.

#### Limit-Existence

If the limit does not exist, then the `x_i` necessarily follow a cyclical path. If we use the entire self-play
dataset as the experience buffer, however, an infinite cyclical path is not possible; we must eventually fall into the path's interior.

(Using the entire self-play dataset as the experience buffer is neither practical not desirable; we consider that
here purely for theoretical arguments.)

(This part of the psuedo-proof is questionable.)

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
