# Connect4

## Perfect solver

To set up external perfect solver:

  1. Clone this repo: https://github.com/PascalPons/connect4
  2. Download the opening book ("7x6.book") from here and copy to that directory: https://github.com/PascalPons/connect4/releases/tag/book
  3. Build the binary via make
  4. In your config.txt at the repo root, add the line "c4.solver_dir = DIR", where DIR is the directory where you cloned the above repo.

## Debug files

We used to have a python mcts implementation, which would produce a debug file that could be visualized through an interactive web tool. We deleted this python code because it was not longer used. At some point, we may add the capability to produce these debug files on the c++ side.

If and when we bring back those debug files, they can be visualized with js machinery:

```
$ cd $REPO_ROOT/js/c4-debug
$ npm install  # one-time
$ npm start
```

This should open up http://localhost:3000/ in your browser. Upload your debug file and enjoy.

