This document explores the possibility of adding an auxiliary value target, that predicts the network's
own `V` prediction for the children of the current node.

## Idea 1: Action Value Target

Currently, for a given node `n`, the network predicts a scalar value `V(n)` and a policy distribution `P(n)`.

Add an auxiliary value-head that predicts a scalar `V_c(n)` for each child `c` of `n`. 
This is a prediction of what the network will predict after each possible move from the current position. 
The targets for this head can be generated during the self-play game by evaluating `V(c)`. The `V(c)` 
value will be available for free for all explored children. For unexplored children, it can be obtained 
cheaply via side-channel evaluation requests.

By itself, this action-value-target might provide a regularizing effect that improves learning,
similarly to how the opponent-response-policy target improves learning. However, we expect this
auxiliary head to have value in other ways - read on!

## Idea 2: Use Action Value Target Predictions Instead of FPU

In PUCT evaluations, when `N(c) == 0`, the `Q(c)` term is undefined. MCTS implementations thus typically employ
an FPU (First Play Urgency) policy that provides a value for this undefined `Q(c)` term. It is known that 
AlphaZero can be quite sensitive to FPU policy, so many projects expend significant effort to tune FPU parameters.

Instead of an unprincipled FPU policy, how about we use `V_c(n)` as the `Q(c)` term when `N(c) == 0`?
This is the principled choice, as it is prediction of what `Q(c)` will be after 1 visit to `c`.

## Idea 3 (Stochastic AlphaZero only): MCTS Backpropagation Denoising

When backpropagating a value during _stochastic_ MCTS, apply an additive chance-correction to the 
value when traversing edges corresponding to chance-events. If the edge connects parent `n` to child `c`,
the chance-correction should be in the amount of `mu - V_c(n)`, where `mu` is the average of `V_c*(n)`
taken over all children `c*` of `n`, with the average taken according to the chance-event's distribution.
This should have a denoising effect on backpropagation mechanics.

For example, suppose we are at a chance node `n` with 3 equally likely children, `c`, `d`, and `e`:

```
   n
  /|\
 c d e
```

Suppose we have the following:

```
V_c(n) = 0.2
V_d(n) = 0.6
V_e(n) = 0.7
```

The average of these is `mu = 0.5`.

Let's say we are backpropagating a value of 0.22 through `c`. The chance-correction here would be:

```
mu - V_c(n) = 0.5 - 0.2 = 0.3
```

So we would backpropagate `0.22` to `c`, but `0.22 + 0.3 = 0.52` to `n` and to `n`'s ancestors.

The idea here is that sampling `c` was bad luck: `0.3` worth of bad luck, to be precise.
It’s unfair to our running measurement of `n`'s quality to penalize it for that bad luck. 
Doing so discourages `n`'s parents from further exploring `n`, even though in this case,
we actually obtained evidence that `c` is _better_ than initially believed, and thus that
`n` may also be better than initially believed. Standard MCTS applies this bad luck penalty,
and this idea removes that penalty.

Note that in the correction term of `mu - V_c(n)`, we can consider replacing `mu` with `V(n)` or
`Q(n)`, and we can consider replacing `V_c(n)` with `V(c)` or `Q(c)`. As `V` and `V_c`
approach perfection, `V(n)` and `mu` should be equivalent, and as `P` approaches perfection,
`Q(n)` should also be equivalent. Similarly, as `V` and `V_c` approach perfection, `V_c(n)` and `V(c)`
should be equivalent, and as `P` approaches perfection, `Q(c)` should also be equivalent.
Our proposed choice of `mu` and `V_c(n)` guarantees that the correction adjustment
is zero-meaned, even if the network is imperfect. Without this zero-mean guarantee, the adjustment
may introduce a bias, and it is unclear what the impact of that bias may be.

## Idea 4 (Stochastic AlphaZero only): Value Target Denoising

Continuing with our above example, the self-play game might reach node `n`, and then `c`, 
ultimately resulting in a final game outcome of `z=0.3`. This result was better than expected at 
`c` (as `V_c(n)` was `0.2`). The value target for `n` should be indicative of this better-than-expected result.
Thus, we should apply that same chance-correction to the value-target for node `n` (and to `n`'s ancestors).

There is a connection between this idea and LeelaChessZero’s deblundering mechanism. With this idea,
when we experience a chance event that causes our expected-result to deviate from expectation, 
we use a model-evaluation to predict the delta that the chance event induced, and correct the 
value target according to that predicted delta. If we think of move-temperature-selection as a chance-event, 
and assume that in competition we would use a temperature of zero, then a deviation from the 
zero-temperature-move is the same thing: experiencing a chance event that causes our expected-result to 
deviate from expectation. Deblundering similarly predicts the delta that the chance event induced and 
corrects the value target according to that predicted delta. A difference is that deblundering uses 
`Q` for this prediction while this idea uses an action-value. Again, as suggested above, we can consider
replacing `V_c(n)` with `Q(c)`, which represents a sort of unification of the ideas.

## Idea 5 (Stochastic AlphaZero only): Action Value Target Regularization at Chance Nodes

If the network is consistent with itself at chance nodes, the value at the parent should equal the 
expected value of the children. The aciont value head predicts the value of each child. This 
allows us to add a loss term that encourages this equality:

```
(V(n) - E_c[V_c(n)])^2
```

where `E_c` is an expectation taken with respect to the chance node's distribution for `c`.

## Idea 6 - Action Value Target Regularization at Non-Chance Nodes

At non-chance nodes, there is a different form of consistency that we can expect.
The value of the parent should equal the value of the child representing the opponent's
best response. We can add a loss term that encourages this equality:

```
(V(n) - max_c V_c(n))^2
```

where `max_c` is a maximum over all children `c`.
