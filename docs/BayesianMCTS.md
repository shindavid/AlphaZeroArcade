# Bayesian MCTS

This document sketches out an alternative to MCTS.

## A Different Formulation of Standard MCTS

AlphaZero's MCTS operates by repeatedly descending a tree, updating various node statistics along the way.

[Grill et al](https://arxiv.org/abs/2007.12509) provided an interpretation of its mechanics: at each node,
we maintain a policy. That policy is initialized with a prior, and repeatedly sharpened based on evidence observed from
its descendents.

It is instructive to rigorously describe MCTS mechanics in terms of this interpretation. We do so
using python syntax:

```
Policy = np.ndarray  # array of floats

@dataclass
class Stats:
  P: Policy      # prior policy, comes from neural net
  pi: Policy     # posterior policy, initialized to self.P
  Q: float       # quality estimate
  N: int         # visit count


@dataclass
class Node:
  children: List[Node]
  stats: Stats
```

At each step, we start with a parent node, and visit one of its children. The visit of the child produces
_evidence_, which can be summarized as a pair of `Stats`: the child stats _before_ the visit, and the
child stats _after_ the visit. The evidence is used to update the parent stats. The update function
looks like this:

```
class Node:
  ...
  def update(self, child_index: int, before: Stats, after: Stats):
    self.update_posterior(child_index, before, after)
    self.stats.N += 1
    self.stats.Q = sum(self.stats.pi[c] * child.stats.Q for c, child in enumerate(self.children))
```

The last line in the above is a natural computation to perform in a multi-armed
bandit setting: to calculate my expected payout from pulling the lever of a randomly
chosen slot machine, I should compute the average of each slot machine's expected payout, weighted by
the probability that I will choose each machine.

The remaining piece to complete the description is the `update_posterior()` method:

```
class Node:
  ...
  def update_posterior(self, child_index: int, before: Stats, after: Stats):
    if self.stats.N == 0:
      self.stats.pi *= 0  # posterior was initialized to prior, but just throw that away!
      self.stats.pi[child_index] += 1
    else:
      self.stats.pi[child_index] += 1 / self.stats.N

    self.stats.pi /= sum(self.stats.pi)  # normalize
```

This is a very strange way to implement this function! It begs many questions:

- When `self.stats.N == 0`, why do we throw away the prior and set the posterior to a singular distribution? Surely, that choice is not the one that optimally combines the prior with the observed data?
- If `before.Q > after.Q`, that means that our belief of the quality of the visited child _decreased_. Why then, do we _increase_ the policy weight for this child?
- In fact, the method does not even bother looking at `before` or `after` at all! Why not?

## A Better `update_posterior()` Method

Let us build a better `update_posterior()` method.

It will be useful to introduce another statistic: a measure of the _uncertainty_ of `Q`. We will call this `U` (for "uncertainty"). 
Intuitively, we want it to estimate how much change we expect in `Q` if we perform more visits to the node. That is, if $R$
is a random variable corresponding to the amount that `n.stats.Q` will change by on the next $k$ visits to node `n`, we want
`n.stats.U` to correspond to the variance of $R$. We want `U` to be initialized by a neural network estimate, and we want
to derive a precise update rule  for `U` - the details of this should be flushed out later.

Note that for nodes corresponding to terminal game states, we will have `U = 0`.

It is not difficult to see that by checking for `U == 0`, we can have our `update_posterior()` instantaneously collapse `pi`
upon finding forced wins, with minimax-mechanics propagating such collapses up the tree to crisply identify mate-in-N situations.

Let `deltaQ = after.Q - before.Q`. At a high-level, our `update_posterior()` should behave as follows:

- If `deltaQ > 0`, we should _increase_ `pi[child_index]`, with the magnitude of the increase determined by `deltaQ / U`.
- If `deltaQ < 0`, we should _decrease_ `pi[child_index]`, with the magnitude of the decrease determined by `deltaQ / U`.

TODO
