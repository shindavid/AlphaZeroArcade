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

# train connect-4
train_c4 tag *extra:
  ./py/alphazero/scripts/run_local.py -g c4 -t {{tag}} {{extra}}
