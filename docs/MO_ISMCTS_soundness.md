# Multiple-Observer Information Set MCTS

This 2012 [paper](https://eprints.whiterose.ac.uk/75048/1/CowlingPowleyWhitehouse2012.pdf) by Cowling et al
introduces Multiple-Observer Information Set MCTS (MO-ISMCTS).

In MO-ISMCTS, a separate game tree is maintained for each player of the game, and the tree nodes are 
information sets from the corresponding player’s point-of-view (POV). On each MCTS iteration, the hidden 
state is instantiated uniformly randomly. Then, _both_ trees are descended simultaneously, performing action 
selection using the statistics of the active player’s tree.

The uniform random instantiation of hidden state is one problem, as it neglects the fact that past actions
inform one's beliefs about the hidden state.

The subtler problem is that this setup permits an inappropriate "leakage" of private information across trees.

To illustrate, suppose Alice and Bob are the two players playing Scrabble, with Alice to act at the root.
In this board state, the "X" tile is a critical tile, and Alice happens to hold it. Bob of course does not know this.
Suppose MO-ISMCTS selects a non-X move for Alice on her turn, to arrive at a node, `B`, for Bob’s turn. On the
children of `B`, we are back at Alice's turn. At these children of B, MO-ISMCTS dictates that we use Alice's tree, for which we Alice's
rack contains the known X-tile. The selection step will choose X-moves often, and this will be reflected in the
backpropagated values back to B. PUCT is likely to favor blocking X-plays at B, with a degree of certainty that
Bob should not have. The fact that Alice has an "X" has "leaked" to Bob.
