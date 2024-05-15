# Luck Adjustments

This document describes technical details of luck-adjustments. See [here](Nash-ISMCTS.md) for context.

## Current Proposal

At each node $n$, we have a value model, $V(n)$, and an aux-value model, $VC(n)$. The latter returns
an array of value-predictions, one for each child of $n$. For notational convenience, for an evaluated parent node $n$ with child node $c$, let:

```math
V^*(c) = \begin{cases}
V(c) & \text{ if } c \text{ has been evaluated} \\
VC(n)[c] & \text{ else}
\end{cases}
```

$V^*(c)$ represents our best-static-evaluation of $c$.

At sampling nodes, we have a _trusted_ sampling distribution $H$. It is trusted in the sense that we directly sample from it to
produce a visit distribution $N$, meaning that we can trust that any differences between $H$ and $N$ must purely be
due to sampling noise. We define:
```math
Q(n) = \bigg(\bigcup_{H' \in B_\epsilon(H)} \mathbb{E}_{c \sim H'} [V^*(c)]\bigg) + \mathbb{E}_{c \sim N}[Q(c) - V^*(c)]
```
Here, $B_\epsilon(H)$ represents an L1-_ball_[^1] of radius $\epsilon$, lying on the simplex, centered at $H$.

One can think of the first term of the sum as a static base-model evaluation, and the second term as a dynamic modeled residual.

Note that if we set $\epsilon = 0$ and assume that $N \equiv H$, then this simplifies to $Q(n) = \mathbb{E}_{c \sim N}[Q(c)]$,
matching standard MCTS. Also note that if our $V$ model is perfect, then the residuals should be zero.

At action nodes, we have an _untrusted_ action distribution $P$. It is untrusted because the visit distribution $N$
is _not_ produced by directly sampling $P$. Instead, the visits come from an implicit refined policy $\hat{P}$ that
a spawned MCTS operator attempts to learn in an online manner. We can similarly define:
```math
Q(n) = \bigg(\bigcup_{P' \in B_\epsilon(\hat{P})} \mathbb{E}_{c \sim P'} [V^*(c)]\bigg) + \mathbb{E}_{c \sim N}[Q(c) - V^*(c)]
```

Unfortunately, $\hat{P}$ is unknown. We therefore make an attempt to model it based on our prior belief $P$
and our observed $N$. The principled approach to do this in a Bayesian manner is to use a Dirichlet distribution,
using $P$ as the prior.

Alternatively, if we want $\hat{P}$ to exactly equal $P$ in a "sticky" manner until enough statistical evidence
emerges to the contrary, we can define:

```math
\hat{P} = \underset{\pi\in SN(N)}{\text{argmin }} d(P, \pi)
```
Here, $d$ is a distance metric, and $SN(N)$ is a _statistical neighborhood_ function, mapping a given distribution $N$ to a set of
distributions which are not statistically unlikely to produce $N$ via sampling.

## Previous Proposals

Initially, for sampling nodes, we proposed:
```math
Q(n) = \bigcup_{H' \in B_\epsilon(H)} \mathbb{E}_{c \sim H'} [Q(c)]
```

This adjusts for luck well, but if $N$ only represents a sparse sample of $H$, then the visit evidence
barely modifies $Q(n)$. In general, one can estimate a population well via polling with only $O(1)$ samples,
and this approachs fails to take advantage of this polling principle.

Next, we proposed a backpropagation-based formulation, where $Q(n)$ is maintained as a running
average of luck-adjusted backpropagated values. Each luck term is computed based on $H'$,
using $Q$-snapshots at the time of backpropagation.
This takes advantage of the polling principle, but the backpropagation formulation effectively
computes an expectation with respect to $N$, with isn't really sensible when we have a
trusted distribution $H'$. Furthermore, the usage of $Q$-snapshots for
luck-correction is suboptimal, as $Q$ estimates get more refined over time.

Next, we proposed:
```math
Q(n) = \bigcup_{H' \in B_\epsilon(H)} \mathbb{E}_{c \sim H' * (N > 0)} [Q(c)],
```
Here, $H' * (N > 0)$ represents a mask of $H'$, zeroing out entries where $N$ is 0.

This takes advantage of the polling principle, by masking $H'$ based on where we have actual visits, and
fixes the incorrect weighting by $N$ in the expectation. However, the luck-correction element is lost;
if by chance we only visited children with high $Q$ values, we end up with an inflated $Q(n)$ estimate.
Various attempts to similarly incorporate a luck term cause the expression to collapse.

[^1]: Previously, we used the term "neighborhood", but are switching to the term "ball" to avoid notational collisions with "N".
