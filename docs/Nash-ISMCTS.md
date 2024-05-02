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

**Pseudo-Theorem 1**: If we start with a policy $\pi$, and repeatedly refine $\pi$
by updating it towards its best local response, then $\pi$ will converge to a Nash Equilibrium,
if the update is performed appropriately.

---

We are intentionally vague about the details here, as the mechanics of Nash-ISMCTS will differ from CFR,
but rely on this same underlying pseudo-theorem.

Pseudo-Theorem 1 applies to CFR if we think of the average policy $\overline{\pi}_n$ in CFR as the policy $\pi$
in Pseudo-Theorem 1.

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

In particular, the game of No-Limit Texas Hold'em (NLHE) has a
small $h$ (each player only has $\binom{52}{2}$ possible hands) but a large $p$,
as the number of possible betting histories grows exponentially with stack size. CFR succeeded in a
simplified version of the game where the number of possible betting histories was artificially constrained,
while ReBeL succeeded in the unconstrained version.

Still, the $h$ and $b$ factors remain. For the games of Scrabble, Hearts, and Stratego, these factors
are prohibitively large, making them still out of reach for ReBeL.

In our proposed framework, Nash-ISMCTS, we will similarly exploit Pseudo-Theorem 1, but rely on
Monte Carlo Tree Search (MCTS) as the best-local-response-computing search routine. As we will see,
this will allow us to remove the linear dependence on $h$ and $b$ in the system memory requirements,
which will finally put those large games within reach.

### Policy Collapse

We would like to note a general challenge that occurs when attempting to apply search at test time
in imperfect information games: _policy collapse_.

For motivation, consider the simple game of Rock-Paper-Scissors. The equilibrium strategy of this 
game is the uniform random strategy $\pi_u$. We can imagine a framework that models the opponent
as using policy $\pi_u$, and computes an optimal response at test time via a best-local-response-computing
routine, such as CFR. The optimal response to $\pi_u$ is itself $\pi_u$.

However, imagine that our model of our opponent is slightly off. Perhaps we model the opponent as using
$\pi_r$, a policy that is close to $\pi_u$ but slightly favors throwing Rock. Or, as using $\pi_s$, a policy
that is close to $\pi_u$ but slightly favors throwing Scissors. The optimal response to either of these
policies is a pure strategy (always throw Paper against $\pi_r$, and always throw Rock against $\pi_s$).
This is the phenomenon of policy collapse: a slight perturbation in an opponent model can cause a search
routine to collapse towards an easily exploitable policy.

**Any sound framework for imperfect information games must be robust against policy collapse.**

Each iteration of CFR performs a best-local-response-computation, and that computation is not robust
against policy collapse. When performing CFR on Rock-Paper-Scissors, each iteration can generally
produce a deterministic policy, as illustrated above. The robustness of CFR as a whole comes from
the averaging of these policies.

ReBeL uses a specialized mechanism to prevent policy collapse. A description of the mechanism is 
beyond the scope of this document, but can be read about in Section 6 of [Brown et al, 2020](https://arxiv.org/pdf/2007.13544).

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
player's private information.

Training $H$ takes some care. A natural approach is to train $H$ over the same
experience buffer used to train $P$, using the
actual values of the hidden state as training targets. This should result in an unbiased $H$
as long as the self-play games are played according to the policy $P$. As we will see later,
if $P$ is near Nash Equilibrium, our root node action selection mechanics will approximately
enforce that self-play games are played according to $P$.

However, this natural approach does not work in practice. The problem is that we want $H$ to converge
to $B(P)$, but the natural approach instead leads to $H$ converging to $B(\overline{P})$. Here,
$B$ is the Bayes' Rule function, and $\overline{P}$ is the average of the policies used over the
entire experience buffer. It is possible for $P$ to fluctuate over the course of the experience buffer,
but for $\overline{P}$ to remain static, leading to a static $H$.

To remedy this, when we obtain the $n$'th policy model, $P_n$, after the $n$'th generation of
training, we use $P_n$ to generate a set, $S_n$, of self-play games, and train a hidden state model, $H_n$,
_only_ using $S_n$. In order to make this technique practical, we apply the following tricks
to speed it up:

1. Generate the games of $S_n$ by sampling directly from $P_n$, without using tree search.
2. Warm-start the training of $H_{n+1}$ by initializing its weights to the weights of $H_n$.

If the system has converged, then ISMCTS should act as an identity operator on $P_n$, and the
sequence ${{H_n}}$ should approach a fixed model, justifying these tricks.

## Node Splitting

As described, our system still has a memory requirement with linear dependencies on $h$ (the size
of the hidden state space) and $b$ (the game's branching factor). This is because our $H$ model
outputs a softmax layer of size $h$ to sample the hidden state, and because our $P$ model outputs
a softmax layer of size $b$ to produce an action policy.

Here is how we address this. Rather than representing
the hidden state as a flat distribution over the entire space, we split a
single SAMPLE node into a series of multiple SAMPLE nodes, each generating a piece of the hidden state.
We call this $H$-_splitting_.
This is similar to how an LLM samples sentences one word at a time. 
For an appropriately chosen $H$-splitting scheme, this can shrink the dependence from $O(h)$ to $O(\log{h})$.
In Scrabble, we can generate a hidden set of tiles one letter at a time,
limiting the output of the $H$ network to a size-27 softmax layer.
In Stratego, we can generate hidden piece identities one piece at a time. 

Similarly, we can split a single ACT node into
a series of multiple ACT nodes, by decomposing an action into multiple sub-actions. 
We similarly call this $P$-_splitting_. For an appropriate chosen $P$-splitting scheme, this can
shrink the dependence from $O(b)$ to $O(\log{b})$.
In Scrabble, we can decompose a move into a board location, followed by tile placements.

We should note that $P$-splitting has already been successfully utilized in some large-branching-factor games.
DeepMind used it in their Starcraft-playing system AlphaStar ([Vinyals et al, 2019](https://www.nature.com/articles/s41586-019-1724-z)).
User @RightfulChip in the Engine Programming Discord has described success in their (closed-source) engine _Rusty_ for
playing the high-branching-factor game [Arimaa](https://en.wikipedia.org/wiki/Arimaa).

## Equilibrium-Idempotence

Let us call this ISMCTS variant imbued with AlphaZero mechanics _AlphaZero-ISMCTS_, or A-ISMCTS. This is not yet
our final Nash-ISMCTS.

Suppose we play $g$ self-play games using A-ISMCTS and train $P$, $V$, and $H$ on the resultant set of 
self-play data. Will the policy produced by running A-ISMCTS with the resultant $P$, $V$, and $H$, for $n$ visits,
converge towards Nash Equilibrium, as $g$ and $n$ approach infinity?

Unfortunately, the answer is no, as the resultant dynamic system will suffer from the policy collapse phenomenon
described earlier. However, in analyzing some of the properties of this dynamic system,
we can obtain some useful insights that will motivate our more sound version.

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

### Utility Belief Dispersion

The theoretical argument above demonstrates that if we are at equilibrium, then A-ISMCTS will
converge towards an idempotent operator. However, we must be _exactly_ at equilibrium. If $V$ or $H$ are off by
even the tiniest of margins, then the idempotence proof fails, and as the number of MCTS iterations approaches
infinity, the visit distribution can collapse to a single point, rather than converge to $P$ as required.
In short, A-ISMCTS is susceptible to policy collapse.

Practically, the number of visits is finite, and so if the networks are close enough, then perhaps the visit
distribution will be close enough to $P$ to make everything work in practice, at least during self-play, when
the number of visits is small and controlled. However, this is not satisfactory. One should
have confidence that the quality of the posterior policy will increase as a function of the number of visits. Without
such a guarantee, it is difficult to trust that AlphaZero will result in long-term improvement. Even if near-equilibrium
networks are produced, it is difficult to trust that policy collapse won't happen at test time, due to using an
excessive number of visits.

Our high-level strategy to remedy this problem is to change the utility belief, $Q$, at each node
of the MCTS tree from a singular _value_, $q \in \mathbb{R}$, to an _interval_, $[a, b] \subset \mathbb{R}$. We
name the mechanism that produces these intervals, _utility belief dispersion_, or UBD.

<table align="center">
  <tr>
    <td><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/Prism_rainbow_schema.png" alt="Prism Dispersion" style="width:100%; max-width: 600px;"/></td>
  </tr>
  <tr>
    <td style="text-align: center;">In optics, dispersion widens a singular frequency into a range of frequencies.</td>
  </tr>
</table>

The dispersion of the singular value to an interval stems from an injection
of uncertainty about the accuracy of $H$. The nature of the uncertainty injection will
prevent the interval widths from shrinking towards zero as the number of visits approaches infinity.
The PUCT formula will then yields PUCT intervals, and the selection criterion will require comparing intervals.
We will treat overlapping PUCT intervals as "tied", and break ties based on $P$.
If the network believes two actions to be approximately equal
in value, this approach can lead to PUCT intervals that remain overlapping in the infinite limit, and
the tiebreaking rule can then lead to the required convergence to $P$.

**Utility Belief Dispersion is our solution to policy collapse.**

With the high-level strategy outlined, let us fill in details.

### SAMPLE nodes: Selection and Backpropagation

Suppose that during descent, we reach a SAMPLE node $n$. Due to $H$-splitting, $n$ might belong to
a chain of SAMPLE nodes, but that will not affect our description. We evaluate $H$ to obtain a sampling distribution $h$, which we
sample from to select a child $c$. We descend to $c$, and after potentially more descent from there,
obtain a utility value $x$ to backpropagate from $c$ to $n$. Suppose for now that $x$ is a singular
value, and that the $Q$ values at the children of $n$ are singular values.

Intuitively, if $x$ was better than expected, we want $Q(n)$ to increase, to encourage further
exploration of $n$. Conversely, if $x$ was worse than expected, we want $Q(n)$ to decrease.
We can achieve this by applying an adjustment $x \mapsto x - \phi(h)$, where,

```math
\phi(h) = Q(c) - \mathbb{E}_{c' \sim h}[Q(c')]
```

Now, we _disperse_ this adjustment term by injecting uncertainty about $h$. Suppose $n$ has $k$
children. Then $h$ is a size $k$ discrete probability distribution, which maps to a point in $\Delta_k$,
the $k$-simplex. Rather than assuming that the
hidden state distribution is _exactly_ $h$, we instead assume that the hidden state distribution lies
somewhere in $N_\epsilon(h)$, the set of points of $\Delta_k$ whose L1-distance from $h$ is bounded by
some fixed parameter $\epsilon > 0$. This produces a _range_ of adjustment terms:

```math
\Phi(h) = \bigcup_{h' \in N_\epsilon(h)} \phi(h')
```

Adding the adjustment range $\Phi(h)$ to $x$ results in a utility interval, meaning that node $n$
accumulates, through backpropagation, a collection of utility intervals.
The $Q$ interval at $n$ is then simply maintained as $[q_{\mathrm{min}}, q_{\mathrm{max}}]$, where
$q_{\mathrm{min}}$ is the average of the interval minimums, and where $q_{\mathrm{max}}$ is the
average of the interval maximums.

We can now relax our assumption that $x$ was a singular value, and that $Q$ was a singular
value at every child of $n$. If $x$ is an interval, and if $Q$ is an interval at the children of
$n$, we simply revise $\phi$ to be the set of possible values that could arise by choosing
a specific value from each of those intervals.

### Non-root ACT nodes: Selection and Backpropagation

If a non-root ACT node has SAMPLE nodes as children, the PUCT formula will operate on $Q$ intervals,
and thus produce PUCT intervals. Here is how we adjust the selection mechanics:

- If one PUCT interval is strictly greater than the others, then deterministically select the corresponding action.

- Otherwise, the PUCT interval whose left endpoint is largest intersects 
$m\geq1$ other intervals. Choose randomly among the actions corresponding to those $(m+1)$ intervals, with each such action $a$
chosen with probability proportional to the _raw_ prior $P(a)$. We call this the _mixing distribution_.
We emphasize _raw_ here to specify that if perturbations like softmax temperature or
Dirichlet noise are applied, we do _not_ want them to influence the mixing distribution.

When backpropagating upwards from a non-root ACT node $n$,
we wish to control for the randomness of the mixing in the overlapping-intervals case.
In normal MCTS, we have:

```math
Q(n) = \mathbb{E}_{a \sim N} [Q(a)]
```

In order to control for mixing randomness, we instead use:

```math
Q(n) = \frac{n_\mathrm{mixed}\cdot\mathbb{E}_{a \sim \mathrm{MIX}}[Q(a)] + n_\mathrm{pure}\cdot\mathbb{E}_{a \sim PN}[Q(a)]}{n_\mathrm{mixed} + n_\mathrm{pure}}
```
Here:

- $n_\mathrm{mixed}$ is the number of mixed actions performed at $n$.
- $n_\mathrm{pure}$ is the number of pure actions performed at $n$.
- $PN$ is the pure action visit distribution.
- $\mathrm{MIX}$ is the average of the $n_\mathrm{mixed}$ mixing distributions used so far.

Note that the node that we backpropagate towards might live in a different ISMCTS tree (if the
current tree was spawned at this point). In this case, the child $Q$ values must come from
the original tree, while the pure/mixed action counts/distributions must come from the
spawned tree.

### Root ACT nodes: Selection

If our networks are near-equilibrium, then Nash-ISMCTS will produce a near-equilibrium posterior policy
$\pi$ at the root. In order for the AlphaZero loop to work properly, we must select according to $\pi$
at the root. Otherwise, the $V$ and $H$ targets will not be consistent with $P$.

Fixing the move temperature to 1 ensures this. However, a move temperature of 1 fails to trim
exploration-induced visits. We will get the right mixing frequencies between the optimal actions, but
have too much weight on low-quality actions. This dilutes expected action quality, both during self-play
and at test time.

We therefore need a more sophisticated move selection scheme - one that acts like a temperature=1 scheme
among the optimal actions, but which acts like a low-temperature scheme for the low-quality actions.

To this end, we adapt the Lower Confidence Bound (LCB) mechanism. In LCB,
the final $Q(a)$ is combined with its associated $N(a)$ to produce a confidence-interval
around $Q(a)$, of the form $I = [Q - \sigma(N), Q + \sigma(N)]$, for some
decreasing positive-valued function $\sigma$. The action $a$ whose lower bound $\mathrm{min}(I(a))$ is
greatest is identified, and all actions $b$ such that $I(b)$ is strictly less than $I(a)$ are
discarded. Only the remaining actions are candidates for move selection.

In our case, our $Q$ utilities are intervals, so our confidence interval takes the form,

```math
I = [\mathrm{min}(Q) - \sigma(N), \mathrm{max}(Q) + \sigma(N)]
```

With this alternate interval definition, we apply the same LCB filtering mechanism. Then, we select
among the remaining actions proportionally to $N$ (i.e., using a temperature of 1).

### Nash-ISMCTS: Final Remarks

This completes the description of Nash-ISMCTS.

Note that belief update dispersion begins at SAMPLE nodes and propagates upwards.
In the absence of SAMPLE nodes, there is no dispersion, and Nash-ISMCTS reduces exactly
to standard MCTS. This is a nice property and is a good reason to
localize the dispersion mechanics to $H$, rather than also applying dispersion to $V$. An
additional reason to avoid applying dispersion to $V$ is that the $N$ term in the PUCT formula
is already motivated by uncertainty in $V$, making additional dispersion potentially redundant.

## Technical Notes

### Child value predictions

(Described in more detail [here](AuxValueTarget.md))

In our description of backpropagation and selection mechanics, we sometimes use
$Q$ estimates for children nodes that have not yet been visited.

To address this, we add a _child-value_ head ($V_c$) to our neural network, alongside the standard value head ($V$).
This produces a vector of values, whose length is equal to the number of children of
the current node $n$. It is trained to predict the network's own evaluation of $V$ at $n$'s children.

The outputs of this head can be used as the default $Q$ value for unvisited nodes.

Note that this helps in standard MCTS as well. In typical AlphaZero implementations, a First Play Urgency (FPU)
mechanism is used to supply $Q$ values for unvisited nodes. The $V_c$ head can be used instead,
and represents a more principled choice.

## TODO

Describe Kuhn poker experiments.
