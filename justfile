repodir := justfile_directory()

# list available recipies
list:
  just --list

# build the repo
build:
  ./py/build.py

# start a docker shell
shell *ARGS:
  ./docker/shell {{ARGS}}

# push repo to a remote machine with rsync
push hostname:
  rsync -avP {{repodir}} {{hostname}}:

# set up a new lambda cloud node
setup-lambda hostname:
  ./cloud/lambda/setup.sh {{hostname}}

goto hostname:
  ssh -t {{hostname}} 'cd AlphaZeroArcade && just shell'

# generate repo configuration file
genconfig:
  #!/bin/bash
  echo "" > config.txt
  echo "#c4.solver_dir = /home/dshin/checkouts/connect4  # connect4 solver dir" >> config.txt
  echo "libtorch_dir = {{repodir}}/extra_deps/libtorch" >> config.txt
  echo "alphazero_dir = {{repodir}}/data" >> config.txt

# train connect-4
train_c4 tag *extra:
  ./py/alphazero/main_loop.py -g c4 -t {{tag}} {{extra}}
