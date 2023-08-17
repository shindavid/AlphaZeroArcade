#!/usr/bin/env python3

"""
Generates a perfect-play lookup table to be pasted into the c++ code.

Does this via brute-force search. Uses symmetries to reduce the search space. Prunes the lookup
table to only include positions that would be encountered when one side is playing perfectly.

The lookup table describes a mixed-strategy, where the probability of playing each move is
proportional to the probability of a win against a randomly playing opponent.
"""
import numpy as np
from typing import List, Optional, Tuple
import sys

np.set_printoptions(precision=3)

seed = None
if len(sys.argv) > 1:
  seed = int(sys.argv[1])

np.random.seed(seed)


"""
Bit mask representation:

6 7 8
3 4 5
0 1 2
"""
BitMask = int
FULL_BOARD: BitMask = 511


def make_mask(*args: int) -> BitMask:
  return sum(1 << i for i in args)


WINNING_MASKS = (
  make_mask(0, 1, 2),
  make_mask(3, 4, 5),
  make_mask(6, 7, 8),
  make_mask(0, 3, 6),
  make_mask(1, 4, 7),
  make_mask(2, 5, 8),
  make_mask(0, 4, 8),
  make_mask(2, 4, 6)
)


def winning(mask):
  return any(mask & winning_mask == winning_mask for winning_mask in WINNING_MASKS)


class Player:
  X = 0
  O = 1


class Outcome:
  UNDECIDED = -1
  X_WIN = 0
  O_WIN = 1
  TIE = 2


class Node:
  cache = {}

  @staticmethod
  def clear_cache():
    Node.cache = {}

  def __init__(self, x_mask: BitMask, o_mask: BitMask, current_player: Player):
    self.masks = [x_mask, o_mask]
    self.full_mask = x_mask | o_mask
    self.current_player = current_player
    self.children = [None] * 9
    self.move_probs = np.zeros(9)
    self.outcome_probs = np.zeros(3)

  def move_number(self) -> int:
    return bin(self.full_mask).count('1')

  def dump(self):
    text = ("6 7 8  | | | |\n" +
            "3 4 5  | | | |\n" +
            "0 1 2  | | | |")

    text = list(text)

    offset_table = (38, 40, 42, 23, 25, 27, 8, 10, 12)
    for i, offset in enumerate(offset_table):
      submask = 1 << i
      if self.o_mask & submask:
        text[offset] = 'O'
      elif self.x_mask & submask:
        text[offset] = 'X'

    print(''.join(text))
    print('move_number:', self.move_number())
    print('moves:', self.move_probs)
    print('outcome:', self.outcome_probs)

  def is_trivial(self):
    """
    A position is trivial if any of the following are true:

    1. The game is over
    2. There is only one legal move
    3. Some square on the board represents an immediate win for either player.
    """
    for mask in WINNING_MASKS:
      num_x = bin(self.x_mask & mask).count('1')
      num_o = bin(self.o_mask & mask).count('1')
      if num_x + num_o >= 2 and num_x * num_o == 0:
        return True

    return bin(self.full_mask).count('1') >= 8

  def get_cpp_ints(self) -> Optional[Tuple[int, int]]:
    """
    If this node is trivial, returns None.

    Else, returns a compact pair of int64's that encodes this board position along with the move
    policy.

    The first int64 contains the x_mask in the high 32 bits and the o_mask in the low 32 bits.

    The second int64 contains the move probability, multiplied by 255 and rounded down to the
    nearest int, for moves 1-8, one byte per move, with move 1 in the high byte and move 8 in the
    low byte. The move probability for move 0 is 1 minus the sum of the other move probabilities.
    """
    if self.is_trivial():
      return None
    move_dict = {m: p for m, p in enumerate(self.move_probs) if p > 0}
    assert move_dict

    first = (self.x_mask << 32) | self.o_mask
    second = 0
    for p in self.move_probs[1:]:
      second <<= 8
      second += int(p * 255)

    return (first, second)

  @property
  def x_mask(self):
    return self.masks[Player.X]

  @property
  def o_mask(self):
    return self.masks[Player.O]

  def needs_expansion(self) -> bool:
    return sum(self.outcome_probs) == 0

  def expand_child(self, move: int) -> 'Node':
    x_mask = self.x_mask | (1 << move) if self.current_player == Player.X else self.x_mask
    o_mask = self.o_mask | (1 << move) if self.current_player == Player.O else self.o_mask

    node = Node.cache.get((x_mask, o_mask), None)
    if node is None:
      node = Node(x_mask, o_mask, 1 - self.current_player)
      Node.cache[(x_mask, o_mask)] = node

      cur_player_mask = x_mask if self.current_player == Player.X else o_mask
      if winning(cur_player_mask):
        node.outcome_probs[self.current_player] = 1
      elif node.full_mask == FULL_BOARD:
        node.outcome_probs[Outcome.TIE] = 1

    self.children[move] = node
    return node

  def legal_moves(self) -> List[int]:
    return [i for i in range(9) if not (self.full_mask) & (1 << i)]

  def random_expand(self):
    legal_moves = self.legal_moves()
    for move in legal_moves:
      child = self.expand_child(move)
      if child.needs_expansion():
        child.perfect_expand()
        assert not child.needs_expansion()

      self.move_probs[move] = 1
      self.outcome_probs += child.outcome_probs

    self.move_probs /= sum(self.move_probs)
    self.outcome_probs /= sum(self.outcome_probs)

  def perfect_expand(self):
    for move in self.legal_moves():
      child = self.expand_child(move)
      if child.needs_expansion():
        child.random_expand()
        assert not child.needs_expansion()

    cp = self.current_player
    self.perfect_expand_helper(  # win certain
      lambda child: child.outcome_probs[cp]==1, lambda child: 1)
    self.perfect_expand_helper(  # win possible, non-loss certain
      lambda child: child.outcome_probs[1-cp]==0, lambda child: child.outcome_probs[cp])
    self.perfect_expand_helper(  # tie certain
      lambda child: child.outcome_probs[Outcome.TIE]==1, lambda child: 1)
    self.perfect_expand_helper(  # loss possible
      lambda child: True, lambda child: 1-child.outcome_probs[1-cp])
    self.perfect_expand_helper(  # loss certain
      lambda child: True, lambda child: 1)

    assert sum(self.move_probs) > 0
    self.move_probs /= sum(self.move_probs)

    for m, p in enumerate(self.move_probs):
      if p > 0:
        self.outcome_probs += p * self.children[m].outcome_probs
    assert sum(self.outcome_probs) > 0
    self.outcome_probs /= sum(self.outcome_probs)

  def perfect_expand_helper(self, cond, weight):
    if sum(self.move_probs) > 0:
      return

    for c, child in enumerate(self.children):
      if child is not None and cond(child):
        self.move_probs[c] = weight(child)

  def make_move(self) -> 'Node':
    if sum(self.move_probs) == 0:
      return None
    return self.children[np.random.choice(9, p=self.move_probs)]

  def get_normalization_candidates(self):
    return [
      (self.x_mask, self.o_mask),
      (rotate90(self.x_mask), rotate90(self.o_mask)),
      (rotate180(self.x_mask), rotate180(self.o_mask)),
      (rotate270(self.x_mask), rotate270(self.o_mask)),
      (reflect(self.x_mask), reflect(self.o_mask)),
      (reflect_and_rotate90(self.x_mask), reflect_and_rotate90(self.o_mask)),
      (reflect_and_rotate180(self.x_mask), reflect_and_rotate180(self.o_mask)),
      (reflect_and_rotate270(self.x_mask), reflect_and_rotate270(self.o_mask)),
      ]

  def normalized_key(self):
    """
    Returns a comparable key that is invariant to symmetries.
    """
    return min(self.get_normalization_candidates())

  def key(self):
    return (self.x_mask, self.o_mask)


def rotate90(mask: BitMask) -> BitMask:
  return ((mask & 0b000000001) << 6) | \
         ((mask & 0b000000010) << 2) | \
         ((mask & 0b000000100) >> 2) | \
         ((mask & 0b000001000) << 4) | \
         ((mask & 0b000010000)     ) | \
         ((mask & 0b000100000) >> 4) | \
         ((mask & 0b001000000) << 2) | \
         ((mask & 0b010000000) >> 2) | \
         ((mask & 0b100000000) >> 6)

def rotate180(mask: BitMask) -> BitMask:
  return rotate90(rotate90(mask))

def rotate270(mask: BitMask) -> BitMask:
  return rotate90(rotate90(rotate90(mask)))

def reflect(mask: BitMask) -> BitMask:
  return ((mask & 0b000000001) << 2) | \
         ((mask & 0b000000010)     ) | \
         ((mask & 0b000000100) >> 2) | \
         ((mask & 0b000001000) << 2) | \
         ((mask & 0b000010000)     ) | \
         ((mask & 0b000100000) >> 2) | \
         ((mask & 0b001000000) << 2) | \
         ((mask & 0b010000000)     ) | \
         ((mask & 0b100000000) >> 2)

def reflect_and_rotate90(mask: BitMask) -> BitMask:
  return reflect(rotate90(mask))

def reflect_and_rotate180(mask: BitMask) -> BitMask:
  return reflect(rotate180(mask))

def reflect_and_rotate270(mask: BitMask) -> BitMask:
  return reflect(rotate270(mask))


def crawl(root: Node, player):
  cache = {}

  queue = [root]
  while queue:
    node = queue.pop(0)
    key = node.key()
    nkey = node.normalized_key()
    if key in cache:
      continue

    cache[key] = node
    if node.current_player == player and key == nkey:
      pair = node.get_cpp_ints()
      if pair is not None:
        yield pair

    for m, p in enumerate(node.move_probs):
      if p > 0:
        queue.append(node.children[m])


def main():
  table = []

  root = Node(0, 0, Player.X)
  root.perfect_expand()
  table.extend(crawl(root, Player.X))

  Node.clear_cache()
  root = Node(0, 0, Player.X)
  root.random_expand()
  table.extend(crawl(root, Player.O))

  for first, second in table:
    print('    lookup_map[0x{:016x}ULL] = policy_t(0x{:016x}ULL);'.format(first, second))


if __name__ == '__main__':
  main()
