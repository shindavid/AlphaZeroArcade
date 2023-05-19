# Connect4

## Edax solver

To set up external perfect solver:

  1. Clone this repo: https://github.com/abulmo/edax-reversi
  2. Build the binary via make
  3. In your config.txt at the repo root, add the lines,

     ```
     othello.edax_dir = DIR
     othello.edax_bin = BIN
     ```

     Here, DIR is the directory where you cloned the above repo, and BIN is the relative path of the compiled binary ("bin/lEdax-x64-modern" if compiled on x64 linux)

