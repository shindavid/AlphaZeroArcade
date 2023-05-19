
# Alpha Zero Arcade

An Arcade for Ancillary Alpha

## Basics

A command-runner called `just` is used for various scripting.

Install instructions are here: https://github.com/casey/just#packages

To view available commands, just run: `just`

## Docker and Cloud GPUs

Some instructions for building and running on GPUs in the cloud

(Note: this was only tested on the lambdalabs cloud, but is easily extensible to others)

Steps:
  1. Create instance and setup ssh config for HOSTNAME
  2. Run `just setup-lambda HOSTNAME` to configure node, build docker container, install all deps (takes about 5-10 minutes)
  3. Run `just goto HOSTNAME` to log into the cloud docker container
  4. Run `just build` to build
  5. Run `just train_c4 YOUR_TAG_HERE -S` to train connect-4

