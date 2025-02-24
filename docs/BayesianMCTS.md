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
a scalar, it actually represents the mean of a _distribution_, $D_i$, that represents our belief of the true quality of $c_i$.
In turn, each distribution $D_i$ actually represents a _projection_ of some _joint_ distribution, $J$, expressible as a probability
distribution over $\mathbb{R}^n$. The true quality of the $n$ children is expressible as a point of $x \in R^n$, and this implicit joint
distribution $J$ represents our beliefs about $x$.

The policy $\pi$ can be thought of as our belief of the max coordinate of $x$.

Let us start by analyzing a simple case: suppose $n=2$, and suppose that the distributions $D_1$ and $D_2$ are independent
normal distributions of mean/variance $(\mu_1, \sigma_1^2)$ and $(\mu_2, \sigma_2^2)$, respectively. If we draw $(x_1, x_2)$ from the
joint distribution, then the difference $x_1 - x_2$ is normally distributed with mean/variance $(\mu_1 - \mu_2, \sigma_1^2 + \sigma_2^2)$,
and so the probability that $x_1 > x_2$ is given by:

```math
p = \Phi\left(\frac{\mu_1 - \mu_2}{\sqrt{\sigma_1^2 + \sigma_2^2}}\right)
```

We would thus have $\pi = [p, 1-p]$.

Suppose we receive an information update of the form:

```math
\begin{align}
\mu_1 &\rightarrow \mu^{*}_1 \\ 
\mu_2 &\rightarrow \mu^{*}_2
\end{align}
```

This would update our policy to $\pi = [q, 1-q]$, where:

```math
q = \Phi\left(\frac{\mu^{*}_1 - \mu_2}{\sqrt{(\sigma^{*}_1)^2 + \sigma_2^2}}\right)
```

Equivalently, we could frame this update rule as multiplying $\pi(1)$ by $\alpha$ and then normalizing $\pi$, where

```math
\alpha = \frac{q(1-p)}{p(1-q)}
```

Can we generalize this approach? Unfortunately, when we generalize beyond 2-dimensions, the problem has no known
analytical solution ([source](https://mathoverflow.net/q/153039)). However, perhaps we can simply perform this computation
independently against each sibling and combine the results in some way?

One idea is to compute the $\alpha$ term against each sibling, and then multiply by all of them, followed by a normalization.
Experimentation is needed.

TODO:

- Discuss options for training a head to predict variance.
- Derive update rule for variance beliefs
- Derive a replacement for PUCT by framing the decision of which leaf to expand in terms of which leaf offers the greatest likelihood
  of changing the root policy, based on the curent node stats.
