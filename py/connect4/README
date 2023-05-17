STEPS

* Set up external perfect solver
  * Clone this repo: https://github.com/PascalPons/connect4
  * Download the opening book ("7x6.book") from here and copy to that directory: https://github.com/PascalPons/connect4/releases/tag/book
  * Build the binary via make
  * TODO: take care of above via git-submodule + setup script?

* Set up config:
  * Make a file at repo root called config.txt
  * Add the line "c4.solver_dir = <dir>" to this file, where <dir> is the directory of your above cloned repo

* Debug files
  * We used to have a python mcts implementation, which would produce a debug file that could be visualized through an
    interactive web tool. We deleted this python code because it was not longer used.
  * At some point, we may add the capability to produce these debug files on the c++ side.
  * To visualize the debug file:

      $ cd $REPO_ROOT/js/c4-debug
      $ npm install  # one-time
      $ npm start

    This should open up http://localhost:3000/ in browser, upload your file

