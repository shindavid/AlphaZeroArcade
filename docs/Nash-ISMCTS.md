# Nash Information Set Monte Carlo Tree Search (Nash-ISMCTS)

This document describes _Nash Information Set Monte Carlo Tree Search_. Nash-ISMCTS is our MCTS-variant,
which can be applied in imperfect information games to approximate a game's Nash equilibrium.

## Background

### The Foundation: Counterfactual Regret Minimization (CFR)

_Counterfactual Regret Minimization_, or CFR ([Zinkevich et al, 2007](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf)),
is a foundational technique in incomplete information games. At a high-level, this technique works
by representing a policy $\pi$ as a look-up table mapping every possible information set in the game to
a probability distribution over actions. This table is of size $O(phb)$, where:

* $p$ is the number of public states of the game
* $h$ is the number of hidden states of the game
* $b$ is the average branching factor of the game

CFR iteratively produces a sequence of policies $P = \lbrace \pi_1, \pi_2, \ldots \rbrace$. 
Let $\overline{\pi}_n$ denote the average of the first $n$ elements of $P$.

The iterative step, roughly speaking, computes a new $\pi_{n+1}$ by updating $\overline{\pi}_n$ towards
its locally-optimal counter-strategy. The exhaustive table facilitates the computation for this update.
The linked paper describes the iterative step more exactly and provides a rigorous
proof that $\overline{\pi}_n$ converges to a Nash Equilibrium.

A key high-level takeaway is this:

---

**Pseudo-Theorem 1**: Suppose we have a set, $S$, of policies. If we repeatedly expand $S$ by adding a
new policy obtained by updating the average policy of $S$ towards its best local response,
then the average of the policies in $S$ will converge to a Nash Equilibrium.

---

We are intentionally vague about the details here, as the mechanics of Nash-ISMCTS will differ from CFR,
but rely on this same underlying pseudo-theorem.

### Integrating Search: Recursive Belief-based Learning (ReBeL)

CFR has achieved great success in games with a tractable table size, including many popular variants of poker.
However, the technique is not suitable for games where $O(phb)$ is large, such as Scrabble, Hearts, or Stratego.

Pseudo-Theorem 1 hints at the possibility of integrating a best-local-response-computing search routine
into a Nash Equilibrium computing framework.

This possibility was first realized in _Recursive Belief-based Learning_, or ReBeL ([Brown et al, 2020](https://arxiv.org/pdf/2007.13544)).
At a high level, ReBeL works by approximating best-response payouts with a value neural network, $V$, and
then performing the best local response computation needed to power Pseudo-Theorem 1 via a search
routine performed on a synthetic shallow game tree whose leaf values come from $V$. That
search routine is itself CFR. 

Compared to the full game tree, these synthetic shallow game trees are much smaller. Their size
only depend on $h$ and $b$, while the full game tree also depends on $p$. Thus, while CFR on the full game tree
uses $O(phb)$ memory, CFR on these synthetic shallow game trees only uses $O(hb)$ memory. The removal of the $p$ term is
what justifies the complexity of ReBeL compared to CFR, and is what allows ReBeL to work on games
that were intractable for CFR alone.

Still, the $h$ and $b$ factors remain. For the games of Scrabble, Hearts, and Stratego, these factors
are prohibitively large, making them still out of reach for ReBeL.

In our proposed framework, Nash-ISMCTS, we will similarly exploit Pseudo-Theorem 1, but rely on
Monte Carlo Tree Search (MCTS) as the best-local-response-computing search routine. As we will see,
this will allow us to remove the linear dependence on $h$ and $b$ in the system memory requirements,
which will finally put those large games within reach.

### Many-Tree Information-Set Monte Carlo Tree Search (MT-ISMCTS)

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

Note that the only piece of information that Alice extracts from her simulation of Bob's thinking is
Bob's _action_. The utility that simulated-Bob perceives his action to have will not get averaged into
her own utility beliefs.

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

$H$ will accept the same inputs as $P$ and $V$: the public game history, together with the acting
player's private information. It will be trained on self-play games using the actual values of the hidden state
as training targets. This should result in an unbiased $H$ as long as the self-play games are played according to
the policy $P$. We can approximately enforce this, under the assumption that $P$ has converged,
by being careful with move selection mechanics; more on this later.

In some games, the size of the hidden state space, $h$, can be intractably large. Rather than representing
the hidden state as a flat distribution over the entire space, we can split a
single SAMPLE node into a series of multiple SAMPLE nodes, each generating a piece of the hidden state.
We call this $H$-_splitting_.
This is similar to how an LLM samples sentences one word at a time. In Scrabble, we can
generate a hidden set of tiles one letter at a time, limiting the output of the $H$ network to a size-27 logit layer.
In Stratego, we can generate the hidden piece identities one piece at a time. This removes the linear dependence
on $h$ in the system memory requirements.

Similarly, if the size of the action space, $b$, is too large, we can split a single ACT node into
a series of multiple ACT nodes, by decomposing an action into multiple sub-actions. 
We similarly call this $P$-_splitting_. In Scrabble, we can
decompose a move into a board location, followed by tile placements. This removes the linear dependence on
$b$ in the system memory requirements.

## Equilibrium-Idempotence

Let us call this ISMCTS variant imbued with AlphaZero mechanics _AlphaZero-ISMCTS_, or A-ISMCTS. This is not yet
our final Nash-ISMCTS.

Suppose we play $g$ self-play games using A-ISMCTS and train $P$, $V$, and $H$ on the resultant set of 
self-play data. Will the policy produced by running A-ISMCTS with the resultant $P$, $V$, and $H$, for $n$ visits,
converge towards Nash Equilibrium, as $g$ and $n$ approach infinity?

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
infinity, the visit distribution can collapse to a single point, rather than converge to $P$ as required.

Practically, the number of visits is finite, and so if the networks are close enough, then perhaps the visit
distribution will be close enough to $P$ to make everything work in practice, at least during self-play, when
the number of visits is small and controlled. However, this is not satisfactory. One should
have confidence that the quality of the posterior policy will increase as a function of the number of visits. Without
such a guarantee, it is difficult to trust that AlphaZero will result in long-term improvement. Even if near-equilibrium
networks are produced, it is difficult to trust that policy collapse won't happen at test time, due to using an
excessive number of visits.

Our high-level strategy to remedy this problem is to replace the utility _point_-estimate $Q\in \mathbb{R}$ at each SAMPLE node
with a utility _interval_-estimate $Q = [a, b] \subset \mathbb{R}$, whose width is based on an assumed level uncertainty about the accuracy of $H$.
The nature of this uncertainty means that the interval's width will _not_
shrink towards zero as the number of visits approaches infinity.
The PUCT formula will then yields PUCT intervals, and the selection criterion will require comparing intervals.
We will treat overlapping PUCT intervals as "tied", and break ties based on $P$.
If the network believes two actions to be approximately equal
in value, this approach can lead to PUCT intervals that remain overlapping in the infinite limit, and
the tiebreaking rule can then lead to the required convergence to $P$.

(By contrast, if we associated our uncertainty with $V$ instead of $H$, then the uncertainty of $Q$
would be expected to shrink towards 0 as more observations are backpropagated,
if we assume independent noise with each observation.
And if we _don't_ assume independent noise, then we likely require some sort of covariance modeling,
which seems quite difficult.)

With the high-level strategy outlined, let us fill in details.

Consider a SAMPLE node $n$, whose parent is an ACT node. Due to $H$-splitting, $n$ may be followed
by an additional $(k-1)$ SAMPLE nodes. The act of sampling a full hidden state then corresponds to
sampling a random route down a depth $k$ tree $T$ rooted at $n$.

Suppose we have made $m$ visits to node $n$. On visit $i$, suppose we traversed $T$ via route $R_i$,
backpropagating value $x_i$, so that $Q(n) = \frac{1}{m}\sum_i x_i$.
We are able to compute, from the $H$ evaluations, the sampling probability $p_i$ associated with route $R_i$.

---

We have not yet flushed out the full details of how we want to construct our utility interval based
on $\overrightarrow{x}$ and $\overrightarrow{p}$, although we have several candidate ideas. This
requires some more thought and experimentation.

For now, let us leave this as a black box. We envision that our assumed uncertainty about $H$ will
be parameterized by a constant $\epsilon > 0$, with an $\epsilon = 0$ corresponding to A-MCTS.
Hence, we will refer to the method of constructing our $Q$ intervals at SAMPLE nodes
as the $\epsilon$-_mechanism_.

---

The $\epsilon$-mechanism will produce $Q$ intervals, in turn producing PUCT intervals. Here is how we adjust the selection mechanics:

- If one PUCT interval is strictly greater than the others, then deterministically select the corresponding action.

- Otherwise, the PUCT interval whose left endpoint is largest intersects 
$m\geq1$ other intervals. Choose randomly among the actions corresponding to those $(m+1)$ intervals, with each such action $a$
chosen with probability proportional to the _raw_ prior $P(a)$. We call this the _mixing distribution_.
We emphasize _raw_ here to specify that if perturbations like softmax temperature or
Dirichlet noise are applied, we do _not_ want them to influence the mixing distribution.

This completes the description of Nash-ISMCTS.

## Technical Notes

### Child value predictions

(Described in more detail [here](AuxValueTarget.md))

In our not-yet-described $\epsilon$-mechanism, we might require $Q$ estimates for children nodes that
have not yet been visited.

To address this, we add a _child-value_ head ($V_c$) to our neural network, alongside the standard value head ($V$).
This produces a vector of values, whose length is equal to the number of children of
the current node $n$. It is trained to predict the network's own evaluation of $V$ at $n$'s children.

The outputs of this head can be used as the default $Q$ value for unvisited nodes.

As a bonus, this $V_c$ head provides us with a sound First Play Urgency (FPU) mechanism that represents
an improvement over typical FPU policies used in other AlphaZero implementations.

As we will see in later sections, $V_c$ will be used in other ways for variance reduction during
backpropagation.

### Reducing variance from sampling

(Described in more detail [here](AuxValueTarget.md#idea-3-stochastic-alphazero-only-mcts-backpropagation-denoising))

When backpropagating a value $x$ from a node $v$ to its parent SAMPLE node $u$, we apply an additive correction to
$x$ to control for the luck of sampling $v$ from $u$. The appropriate adjustment is:

```math
Q(v) - \mathbb{E}_{w \sim H(u)} [Q(w)],
```

where $H(u)$ represents the sampling distribution gotten by evaluation $H$ at $u$,
and where the $Q$ values are computed _before_ incorporating $x$. For not-yet-expanded children, we can use $V_c$
in place of $Q$.

We apply this same mechanism at random chance nodes, where the chance event comes from the game rules
(such as random tile draws from the bag in Scrabble).

At action nodes, in the PUCT-interval-overlap case, we also have a sampling event. We apply the same mechanism in this case.

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
around $Q(a)$, of the form $I = [Q - \phi(N), Q + \phi(N)]$, for some
decreasing positive-valued function $\phi$. The action $a$ whose lower bound $\mathrm{min}(I(a))$ is
greatest is identified, and all actions $b$ such that $I(b)$ is strictly less than $I(a)$ are
discarded. Only the remaining actions are candidates for move selection.

In our case, our $Q$ utilities are intervals, so our confidence interval takes the form,

```math
I = [\mathrm{min}(Q) - \phi(N), \mathrm{max}(Q) + \phi(N)]
```

With this alternate interval definition, we apply the same filtering mechanism. Then, we select
among the remaining actions proportionally to $N$ (i.e., using a temperature of 1).

### Bayes' Rule Loss Term

We have argued that $H$ should converge towards the distribution that reflects $P$ combined with Bayes' Rule.
While this is true, it will likely help to add an additional loss term to pressure the $H$ head to conform
to $P$.

To this end, we construct a loss term that penalizes local deviations from Bayes' Rule. During self-play, at a given SAMPLE
node, we will have made $n$ queries to $H$, producing a set of $k \leq n$ distinct sampled hidden states: $s_1, s_2, \ldots, s_k$.
Here, we treat a logical SAMPLE node that was split into a series of sub-nodes as if it were collapsed back into one.

Each $s_i$ has an associated sampling probability, $p_i$.

Let $a$ be the last observed opponent action. For each $s_i$, we can reconstruct the information set $I_i$
prior to observing $a$, and compute terms $q_i = \mathrm{Pr}[I_i] \cdot \mathrm{Pr}[a | I_i]$. By Baye's Rule,
we expect the $p_i$ terms to be proportional to the $q_i$ terms. So we can add a loss term based on the
delta between $p_i$ and $q_i$.

### Online H refinement

The Bayes' Rule calculation described above can similarly be performed online, in order to refine the distributions 
used in the $\epsilon$-mechanism. Specifically, we can replace the probability terms used in the correction
term expectation calculation with terms computed by evaluating $H$ and $P$ on prior states. If the calculation
is tractable, we can take those prior states arbitrarily high in the game tree, even as far as the game's starting
position. For games where it is tractable to do this from the starting position, we don't need an $H$ model at all.

(ReBeL does exactly this, forgoing an $H$ model and instead continuously updating hidden state distributions
via Bayes' Rule throughout the entire game.)

## TODO

Describe Kuhn poker experiments.
