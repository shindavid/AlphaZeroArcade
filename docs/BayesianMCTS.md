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

(Note that we are letting `Q` always be in terms of the first-player's POV, for simplicity of exposition.)

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
- The `after` evidence might show us that a child is provably winning. Why not incorporate such evidence by collapsing the posterior?

## A Better `update_posterior()` Method

The questions raised above suggest that we might be able to improve MCTS by substituting a better `update_posterior()` method.
To do this well, let us lay a strong theoretical foundation.

The parent has $n$ children, $c_1, c_2, \ldots, c_n$. Each child $c_i$ has a corresponding `Q` value, $Q_i$. Although this is
a scalar, it actually represents the mean of a _distribution_, $B_i$, that represents our belief of the true quality of $c_i$.
In turn, each distribution $B_i$ actually represents a _projection_ of some _joint_ distribution, $J$, expressible as a probability
distribution over $\mathbb{R}^n$. The true quality of the $n$ children is expressible as a point of $x^* \in R^n$, and this implicit joint
distribution $J$ represents our beliefs about $x^*$.

The policy $\pi$ can be thought of as our belief of the max coordinate of $x^*$.

If each $B_i$ is independent, then $\pi$ could be directly computed from the $B_i$. And, if each $B_i$ were of a constant shape
uniquely determined by $Q_i$, then $\pi$ could be directly computed from the $Q_i$. However, this is usually far from the case, which
is why we have a `P`-head in the first place.

To summarize, the parent policy and the $n$ child `Q` values can be thought of as _projections_ of some implicit underlying
distribution $J$ over $R^n$. When we get new evidence that alters our child `Q` beliefs, we want to, either implicitly or explicitly,
update our belief of $J$, and then based on that, update our policy $\pi$.

It immediately becomes clear that the change in child `Q` does not by itself provide enough evidence to properly update `pi`.
Something else is needed.

To simplify, let us consider a case when there are only two children. Suppose that currently, we have:

```math
\begin{align}
\pi &= [0.2, 0.8]  \\
Q_1 &= 0.4  \\
Q_2 &= 0.5
\end{align}
```

We then obtain evidence that updates our belief of $Q_1$ from $0.4$ to $0.6$. How should we update $\pi$?

TODO
