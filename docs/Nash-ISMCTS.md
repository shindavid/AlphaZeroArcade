# Nash Information Set Monte Carlo Tree Search (Nash-ISMCTS)

This document describes _Nash Information Set Monte Carlo Tree Search_. Nash-ISMCTS is our MCTS-variant,
which can be applied in imperfect information games to approximate a game's Nash equilibrium.

## Tree Search in Imperfect Information Games

A common objection to applying a tree-search algorithm to games of imperfect information
is that in imperfect information games, you cannot analyze a subtree of the game in isolation. 
This is because the value of a node in the game tree is dependent on the policy that got to that point in the tree.

Well...that’s only _partially_ true. It is true that the value of the node against the 
_worst case_ adversary policy is dependent on the policy that got to that point in the tree. This is because
given a policy $P$, the behavior of its optimal counter-policy, $\mathrm{counter}(P)$,
at a node $n$ is dependent on $P$'s behavior across the _entire_ game-tree, 
rather than just on $P$'s behavior in the subtree $T$ rooted at $n$.
However, against any given _fixed_ adversary policy, the value is independent of the behavior of $P$
outside of $T$. So, if we hold the adversary policy fixed, a tree-search subroutine like MCTS could conceivably still act as a
policy improvement operator.

This begs the question: what good is policy improvement against some given _fixed_ policy? In rock-paper-scissor,
against a fixed 100%-rock opponent, iterated policy improvement will lead you to the 100%-paper strategy. That is
not the policy you want to converge to if you want to do well against other policies.

Despite this cautionary example, the fact is, other proven approaches like CFR and R-NaD operate similarly:
on any given generation, they assume a fixed adversary policy and perform a policy-improvement operator with
respect to that _fixed_ policy. Doing so yields a _sequence_ of policy-pairs, and as the policies in the pair
iteratively improve against each other, this _sequence_ (or, the _average_ of the sequence)
converges towards equilibrium.

So, can MCTS, in the AlphaZero context, similarly serve as a policy-improvement operator that, despite
only representing an improvement against a fixed-opponent-policy _within_ a generation, represents a
long-term improvement towards equilibrium _across_ generations?

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

Let us start at the top left red node labeled $x?$. It is Alice's turn. The node is contained in a red
tree, which indicates that the calculations are from Alice's point-of-view (POV). She performs her action
according to an action selection criterion (ACT). Here, she selects the right action, along the edge
labeled $N_b$.

Alice now wishes to simulate Bob's action, as would typically be the next step in standard MCTS. However,
without Bob's private information, Alice cannot simulate his action. Thus, Alice _samples_ Bob's
private information, according to some belief distribution. Here, she samples that his hidden
information is $z$.

She is now able to simulate Bob's action. However, she must account for the fact that Bob does not know
Alice's private information. She thus _spawns_ a new tree, from Bob's POV. This tree is colored blue, to
indicate that it corresponds to Bob's POV. The root of this node is labeled $?z$, to reflect the fact that
Bob knows his private information and does not know Alice's information. The simulation of Bob's action
recursively repeats the same procedure that we started with.

Bob's recursive application of this procedure can itself spawn a new tree from the already-spawned tree,
represented by the third tree in the diagram. And this recursive spawning can continue indefinitely.

This repeated spawning is important. In Alice's simulation of Bob's thinking, she must conceive of him
as an agent that believes Alice can have non- $x$ hidden-states, optimizing his action accordingly. The
$Q$ values relevant to her decision-making should reflect her own known hidden information of $x$, but
the $Q$ values relevant to Bob's decision-making should not. Furthermore, her simulation of Bob's thinking
is an agent that is simulating her own thinking. The simulated-Alice within Alice's simulation of Bob
must potentially have non- $x$ states, so they cannot use the $Q$ values in the original-tree. And so forth.

As [Cowling et al, 2015](http://orangehelicopter.com/academic/papers/cig15.pdf) predated AlphaGo/AlphaZero,
some of the details of the MCTS mechanics were different from what they would be in a more modern
implementation:

- Leaf values were obtained via random rollout rather than via a neural network evaluation
- The action-selection criterion did not incorporate a policy prior, and used a different formula
from PUCT.

## Hidden State Model

In order to build towards Nash-ISMCTS, we will imbue MT-ISMCTS with AlphaZero-mechanics in the natural way:

- Replace random rollouts with value network ($V$) evaluations
- Incorporate a policy network ($P$) and use PUCT as the action selection criterion.

An important detail omitted in our description of MT-ISMCTS is the SAMPLE step. How should Alice sample Bob's hidden information?
[Cowling et al, 2015](http://orangehelicopter.com/academic/papers/cig15.pdf)
propose a variety of approaches to sample this information, with accompanying experimental results.
Instead of adopting one of their proposed approaches, we will instead use an
AlphaZero-inspired approach: train a _hidden-state_ neural network $H$
that learns to sample the hidden state of the game. Note that in principle, $H$ can be computed exactly from $P$ via
Bayes' Rule, but this computation can be expensive. So $H$ can be considered an alternate representation of $P$ that we
use to avoid an expensive online Bayes' Rule calculation.

$H$ will accept the same inputs as $P$ and $V$, and will be trained on self-play games using the actual values of the hidden state
as training targets. This should result in an unbiased $H$ as long as the self-play games are played according to
the policy $P$. We can approximately enforce this by being careful with move selection mechanics (such as
move temperature), under the assumption that $P$ has converged; more on this later.

In some games, the size of the hidden state space can be intractably large. Rather than representing
the hidden state policy as a flat distribution over the entire space, we can split the
single SAMPLE node into a sequence of multiple SAMPLE nodes, each generating a piece of the hidden state.
This is similar to how an LLM samples sentences one word at a time. In Scrabble, we can
generate a hidden set of tiles one letter at a time, limiting the output of the $H$ network to a size-27 logit layer.
In Stratego, we can generate the hidden piece identities one piece at a time.

## Q calculations

The traditional formulation of MCTS maintains a $Q$ value at each node $n$, which corresponds to the running average of
the utility values sampled in the subtree rooted at $n$.

It turns out that there is an equivalent formulation:

```math
Q(n) = \mathbb{E}_{c \sim C(n)}[Q(c)]
```

Here, $C(n)$ denotes the children of $n$. If $n$ is an action node, the distribution dictating the selection of $c$ from $C(n)$
is the action policy at $n$, which is simply the child visit distribution ($N$). See 
[here](https://github.com/lightvector/KataGo/blob/master/docs/GraphSearch.md) for an excellent derivation.

**We will use this alternate formulation.**

In standard MCTS, all nodes are action nodes, and for action nodes, there is no difference in the formulations.

In ISMCTS, however, some nodes are sampling nodes, for which there _is_ a difference. If the sampling distribution is
30% - 70% between actions $a$ and $b$, the formula will yield $0.3 * Q(a) + 0.7 * Q(b)$, regardless of how many
times we happened to sample $a$ vs $b$. This helps decrease variance by mitigating random noise from $H$-sampling.

Later, we will modify the expectation-computation at action nodes as well, in an effort to mitigate random noise
from indifferent actions.

## Equilibrium-Idempotence

Let us call this ISMCTS variant imbued with AlphaZero mechanics _AlphaZero-ISMCTS_, or A-ISMCTS. This is not yet
our final Nash-ISMCTS.

Suppose we play $G$ self-play games using A-ISMCTS and train $P$, $V$, and $H$ on the resultant set of 
self-play data. Will the policy produced by running A-ISMCTS with the resultant $P$, $V$, and $H$, for $n$ visits,
converge towards Nash Equilibrium, as $G$ and $n$ approach infinity?

Unfortunately, the answer is no, as the resultant dynamic system will exhibit unstable chaotic behavior. However,
in analyzing some of the properties of this dynamic system, we can obtain some useful insights that will motivate
our more sound version.

We start with a theoretically interesting property of the system: if $P$, $V$, and $H$ exactly match Nash Equilibrium,
then the visit distribution $N$ produced by A-ISMCTS will approach $P$ as $n$ approaches infinity. When viewing
MCTS as an operator mapping policies to policies, we can say that A-ISMCTS is idempotent at equilibrium, or that
it exhibits _equilibrium-idempotence_.

To see this, note that the PUCT formula is:

```math
\mathrm{PUCT}(a) = Q(a) + c_{\mathrm{PUCT}} * P(a) * \frac{\sqrt{\sum N}}{1 + N(a)}
```

By standard properties of Nash equilibria, the assumption of equilibrium translates to the following: 
$P(a) = 0$ for all $a \not \in S$, where $S$ is the set of actions $a$ for which $V(a)$ attains it maximum
of $v_{max}$.

If $V$ is accurate, as presumed by the equilibrium hypothesis, then $Q(a)$ should converge
towards $v_{max}$ for all $a \in S$, and should converge towards strictly smaller values for
$a \not \in S$. Since
$Q$ dominates the equation as $N(a) \rightarrow \infty$, the proportion of $N$ on actions not in $S$
should approach 0. It remains to show that the ratio $N(a_i) / N(a_j)$ approaches $P(a_i) / P(a_j)$
for all $a_i, a_j \in S$.

To see this, we can plug $a_i$ and $a_j$ into the PUCT equation, and set PUCT values equal. The $Q(a)$, 
$c_{\mathrm{PUCT}}$, and $\sqrt{\sum N}$ terms all cancel, leaving us with:

```math
\frac{P(a_i)}{P(a_j)} = \frac{1 + N(a_i)}{1 + N(a_j)}
```

This demonstrates the required ratio limit, and thus proves the claimed property of equilibrium-idempotence.

## From AlphaZero-ISMCTS to Nash-ISMCTS

The theoretical argument above demonstrates that if we are at equilibrium, then A-ISMCTS will
converge towards an idempotent operator. However, we must be _exactly_ at equilibrium. If $V$ or $H$ are off by
even the tiniest of margins, then the idempotence proof fails, and as the number of MCTS iterations approaches
infinity, the visit distribution will collapse to a single point, rather than converge to $P$ as required.

Practically, the number of visits is finite, and so if the networks are close enough, then perhaps the visit
distribution will be close enough to $P$ to make everything work in practice. However, this is not satisfactory. One should
have confidence that the quality of the posterior policy will increase as a function of the number of visits. Without
such a guarantee, it is difficult to trust that AlphaZero will result in long-term improvement. Even if near-equilibrium
networks are produced, it is difficult to trust that policy collapse won't happen at test time, due to using an
excessive number of visits.

To remedy this problem, we inject uncertainty into our usage of $H$. At a given SAMPLE node, $H$ produces a
sampling distribution $h$, over $k$ possible hidden states. This can be represented as a point in the 
$k$-simplex, $\Delta_k$. Rather than assuming that $h$ is the exact true sampling distribution, we will
relax our belief and assume instead that the true sampling distribution falls
somewhere in the neighborhood $N_{\epsilon}(h)$, consisting of the points of $\Delta_k$ whose
L1-distance to $h$ is bounded by $\epsilon$.

Using an exact $h$ in the $Q$ calculation at a SAMPLE node yields a calculation producing an exact value $q \in \mathbb{R}$:

```math
Q(n) = \mathbb{E}_{c \sim h}[Q(c)]
```

Relaxing to $N_{\epsilon}(h)$ yields instead an _interval_ $[q_1, q_2] \subset \mathbb{R}$:

```math
Q(n) = \bigcup_{h' \in N_{\epsilon}(h)} \mathbb{E}_{c \sim h'}[Q(c)]
```

At ACT nodes, the PUCT formula now operates on child $Q$ _intervals_ rather than values, in turn
producing PUCT intervals. Here is how we adjust the selection mechanics:

- If one PUCT interval is strictly greater than the others, then select the corresponding action.

- Otherwise, the PUCT interval containing the maximum value of the interval-union intersects
$m\geq1$ other intervals. Choose randomly among the actions corresponding to those $(m+1)$ intervals, with each such action $a$
chosen with probability proportional to the _raw_ prior $P(a)$. We call this the _mixing distribution_.
We emphasize _raw_ here to specify that if perturbations like softmax temperature or
Dirichlet noise are applied, we do _not_ want them to influence the mixing distribution.

If $\epsilon=0$, this reduces to A-MCTS, and incurs the previously described instability.
For $\epsilon>0$, however, as long as $N_{\epsilon}(h)$ contains the true equilibrium
sampling distribution, this relaxation stabilizes the system, leading to convergence towards
the prior $P$ as $n \rightarrow \infty$, and thus providing $\epsilon$-_equilibrium-idempotence_.

This completes the description of Nash-ISMCTS.

## Technical Notes

### Reducing variance from indifferent actions

When PUCT intervals overlap, the acting player is indifferent between two or more actions, believing them to have
nearly identical (exploration-incentive-adjusted) utilities. However, the opposing player may be far from
indifferent with respect to those action choices!

As an example, consider a poker game between Alice and Bob. Alice holds a medium-strength hand and is facing a bet
from Bob. Based on the bet size and her belief about Bob's cards, she finds herself indifferent between the
choices of calling and folding, believing them to have equal expected value. Bob, however, happens to have a bluff
in this specific situation. _He_ is not indifferent to whether Alice calls or folds.

In perfect information games, we do not have this asymmetry, since if Alice is indifferent between two or more choices, Bob
will be equally indifferent.

In Nash-ISMCTS, when we sample a specific action $a$ in the overlapping-intervals case, this increments $N(a)$, which has a
direct impact on the parent $Q$ evaluation:

```math
\begin{align*}
Q(n) &= \mathbb{E}_{c \sim C(n)}[Q(c)] \\
     &= \frac{1}{\sum N}\bigg(N(a)\cdot Q(a) + N(b)\cdot Q(b) + \cdots \bigg) \\
\end{align*}
```

If $Q(a)$ is significantly lower than or higher than the true value of $Q(n)$ (as it would be in this poker example),
this can lead to a significant misestimate of $Q(n)$. This in turn can impact upstream PUCT calculations. In
the infinite limit, this will even out, but in practice, the misestimate can have a large distortive effect that
takes a very long time to self-correct.

To mitigate this problem, we differentiate between _pure_ actions and _mixed_ actions. Pure actions are actions resulting from
the unique-maximal-interval case, while mixed actions are actions resulting from the overlapping-intervals case.
A mixed action is similar to a sampling-event, since it similarly results from the random draw of a fixed
distribution. We can thus treat them similarly for the purposes of calculating $Q(n)$, relying on expectations
rather than the actual sampled choices. This looks like this:

```math
Q(n) = \frac{n_{\mathrm{mixed}}\cdot\mathbb{E}_{a \sim \mathrm{MIX}}[Q[a]] + n_{\mathrm{pure}}\cdot\mathbb{E}_{a \sim N}[Q[a]]}{n_{\mathrm{mixed}} + n_{\mathrm{pure}}}
```

Here:

- $n_{\mathrm{mixed}}$ is the number of mixed actions performed at $n$.
- $n_{\mathrm{pure}}$ is the number of pure actions performed at $n$.
- $N$ is the pure action visit distribution.
- $\mathrm{MIX}$ is a distribution, taken by averaging the $n_{\mathrm{mixed}}$ mixing distributions used so far.

### Child value predictions

(Described in more detail [here](AuxValueTarget.md))

When calculating $Q$ for SAMPLE nodes, we sometimes require $Q$ values from children that may not have been expanded.
Descending to each child and querying $V$ could be costly. Instead, we add
an auxiliary network head, the _child-value_ head, $V_c(n)$. This produces a vector of values, whose length
is equal to the number of children of the current node $n$. It is trained to predict the network's own
evaluation of $V$ at $n$'s children.

Training targets for the children can be obtained during self-play by evaluating $V$ for all children.
For expanded children, this $V$ value is available for free. For other children, they can be obtained
via side-channel batched inference requests that do not block the self-play game.

We can similarly train $V_c$ on ACT nodes. This helps with the modified $Q$ calculation described in the
prior section, which similarly requires $Q$ values for children that may not have been expanded. More than
that, $V_c$ can be used in the standard PUCT formula in place of $Q$ for unvisited nodes. This effectively
serves as an alternate FPU policy, and can be applied in standard MCTS in perfect information games as well.

### Move selection

If our policy prior $P$ is near-equilibrium, then Nash-ISMCTS will produce a near-equilibrium posterior policy
$\pi$. In order for the AlphaZero loop to work properly, the self-play agents must act according to $\pi$.
Otherwise, the $V$ and $H$ targets will not be consistent with $P$.

Fixing the move temperature to 1 ensures this. However, a move temperature of 1 fails to trim
exploration-induced visits. We will get the right mixing frequencies between the optimal actions, but
have too much weight on low-quality actions. This dilutes expected action quality, both during self-play
and at test time.

We therefore need a more sophisticated move selection scheme - one that acts like a temperature=1 scheme
among the optimal actions, but which acts like a low-temperature scheme for the low-quality actions.

To this end, we adapt the Lower Confidence Bound (LCB) mechanism. In LCB,
the final $Q(a)$ is combined with its associated $N(a)$ to produce a confidence-interval
around $Q(a)$, of the form $I(a) = [Q(a) - \sigma(N(a)), Q(a) + \sigma(N(a))]$, for some
decreasing positive-valued function $\sigma$. The action $a$ whose lower bound $\mathrm{min}_I I(a)$ is
greatest is identified, and all actions $b$ such that $I(b)$ is strictly less than $I(a)$ are
discarded. Only the remaining actions are candidates for move selection.

In our case, our $Q(a)$ is already an interval, so our confidence interval takes the form,

```math
I(a) = [\mathrm{min}(Q(a)) - \sigma(N(a)), \mathrm{max}(Q(a)) + \sigma(N(a))]
```

With this alternate interval definition, we apply the same filtering mechanism. Then, we select
among the remaining actions proportionally to $N$ (i.e., using a temperature of 1).

## TODO

Describe Kuhn poker experiments.
