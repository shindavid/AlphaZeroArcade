repodir := justfile_directory()

# list available recipies
list:
  just --list

# build the repo
build:
  ./extra_deps/update.sh
  ./py/build.py

# start a docker shell
shell:
  ./docker/shell

# push repo to a remote machine with rsync
push hostname:
  rsync -avP {{repodir}} {{hostname}}:

# generate repo configuration file
genconfig:
  #!/bin/bash
  echo "" > config.txt
  echo "#c4.solver_dir = /home/dshin/checkouts/connect4  # connect4 solver dir" >> config.txt
  echo "libtorch_dir = {{repodir}}/extra_deps/libtorch" >> config.txt
  echo "eigenrand_dir = {{repodir}}/extra_deps/EigenRand" >> config.txt
  echo "tinyexpr_dir = {{repodir}}/extra_deps/tinyexpr" >> config.txt
  echo "cmake.j = 8" >> config.txt
  echo "alphazero_dir = {{repodir}}/data" >> config.txt

# train connect-4
train_c4 tag *extra:
  ./py/alphazero/main_loop.py -g c4 -t {{tag}} {{extra}}
