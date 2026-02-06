#!/usr/bin/env python3

"""
fork_run.py effectively copies a run-directory to a new directory.

###############
# Basic usage #
###############

./fork_run.py -g GAME -f FROM_TAG -t TO_TAG

The above cmd copies all files/directories from FROM_TAG to TO_TAG. For the file categories that
take up a lot of disk space (self-play data, model files, checkpoint files), we by default merely
set up soft-links from TO_TAG to FROM_TAG. It is the responsibility of the user to make sure not to
delete FROM_TAG if intending to still access those files from TO_TAG.

Beyond the basic usage, there are two other modes which can be useful in some common research
scenarios:

#####################
# Retraining Models #
#####################

Sometimes, you want to experiment with a new NN model architecture, or new NN training
hyperparameters. When doing this, you can essentially treat this as a supervised learning (SL)
problem, rather than a reinforcement learning (RL) problem. That is, you can use the existing
self-play data to train the new model. To determine whether your new architecture/hyperparameters
represent an improvement, you can look at the dashboard and compare the loss-curves. If your new
architecture is better, it should do a better job of fitting the existing self-play data, and this
should be reflected when comparing the loss-curves between the two runs.

Doing it this way should be more efficient than doing a fresh run from scratch, as you get to skip
the self-play part. Additionally, by using the same self-play data (and training windows) as the
original run, you can be more certain that comparing the loss-curves is an apples-to-apples
comparison.

For this mode, you want to add the --retrain-models option:

./fork_run.py -g GAME -f FROM_TAG -t TO_TAG --retrain-models

(NOTE: this cmd still launches the eval-server and performs eval-matches throughout the retraining.
Given the motivating use-case I describe, we should consider tweaking the loop-controller behavior
to only have the eval manager(s) issue requests after the retraining period ends.)

#########################
# Branching a Prior Run #
#########################

Sometimes, you want to experiment with c++ changes, without changing the NN model architecture.
Perhaps you believe that your c++ changes will increase the skill-ceiling of your MCTS agent.

An inefficient way to validate this would be to do a fresh run from scratch, and to compare the
eval curves. For a complex game, getting all the way to the skill-frontier might take several days
of self-play and training, and you would need to incur that full cost.

A better approach is to fork the old run!

You could use the basic-usage cmd line to create an exact fork of the old run. However, it's
unclear what you could conclude from the forked run. To illustrate, suppose your old run ran up to
gen-1000, and you fork it and run your forked run up to gen-1100. You can look at the eval-curve
from gen-1000 to gen-1100. It might increase. But, maybe your old run would have also increased if
you just let it run longer?

Because of this conundrum, a more conclusive experiment would be to fork your old run up to gen-G
for some G < 1000. Say, fork the first 900 generations of your old run. Then, on your forked run,
you can let it run up to gen-1000. And then you can compare the eval curves between the two runs
between gen-900 and gen-1000.

For this usage, you can add the --last-gen option:

./fork_run.py -g GAME -f FROM_TAG -t TO_TAG --last-gen GEN
"""
from alphazero.logic.agent_types import AgentRole, MCTSAgent
from alphazero.logic.arena import Arena
from alphazero.logic.custom_types import Generation
from alphazero.logic.rating_db import RatingDB
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.base_dir import Workspace
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer
import games.index as game_index
from util.logging_util import LoggingParams, configure_logger
from util.index_set import IndexSet
from util import sqlite3_util

import numpy as np

import argparse
import dataclasses
import logging
import os
import re
import shutil
from typing import Dict, Optional


logger = logging.getLogger(__name__)
DB_FILE_REGEX = pattern = re.compile(r'^(.+)\.db$')


def load_args():
    description = """fork_run.py effectively copies a run-directory to a new directory.

By default, this does a "soft fork", meaning that the new run will soft-link to the self-play
data, models, and checkpoints of the previous run. If --hard-fork is specified, those files are
copied.

Noteworthy options:

--last-gen:
    Ignore any data that comes after this generation.

--retrain-models:
    Do not copy model files. The forked run will retrain the models using the same self-play data
    and training-windows as the previous run. This is useful for doing supervised-learning
    experiments, such as trying different network architectures or training hyperparameters.
"""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)

    group = parser

    game_index.add_parser_argument(group, '-g', '--game')
    group.add_argument('-f', '--from-tag', help='tag to fork from')
    group.add_argument('-t', '--to-tag', help='tag to fork to')
    group.add_argument('--hard-fork', action='store_true',
                       help='copies self-play data + models + checkpoints from the previous run. '
                       'By default, the new run will still point to the previous run for self-play '
                       'data.')
    group.add_argument('--last-gen', type=int,
                       help='Only copy models up to this generation, along with data created by '
                       'those models')
    group.add_argument('--retrain-models', action='store_true',
                       help='Do not copy model files. The forked run will retrain the models '
                       'using the same self-play data and training-windows as the previous run.')

    LoggingParams.add_args(parser)

    return parser.parse_args()


def copy_databases(source: DirectoryOrganizer, target: DirectoryOrganizer,
                   retrain_models: bool=False, last_gen: Optional[Generation]=None):
    shutil.copyfile(source.clients_db_filename, target.clients_db_filename)

    if not retrain_models:
        copy_eval_databases(source, target, last_gen=last_gen)

        if last_gen is None:
            if os.path.exists(target.ratings_db_filename):
                shutil.copyfile(source.ratings_db_filename, target.ratings_db_filename)
            shutil.copyfile(source.training_db_filename, target.training_db_filename)
        else:
            if os.path.exists(target.ratings_db_filename):
                sqlite3_util.copy_db(source.ratings_db_filename, target.ratings_db_filename,
                                    f'mcts_gen <= {last_gen}')
            sqlite3_util.copy_db(source.training_db_filename, target.training_db_filename,
                                f'gen <= {last_gen}')

    if last_gen is None:
        shutil.copyfile(source.self_play_db_filename, target.self_play_db_filename)
    else:
        sqlite3_util.copy_db(source.self_play_db_filename, target.self_play_db_filename,
                                f'gen < {last_gen}')  # NOTE: intentionally using <, not <=


def tag_from_db_filename(db_filename: str) -> Optional[str]:
    m = DB_FILE_REGEX.fullmatch(db_filename)
    if m is None:
        return None
    return m.group(1)


def copy_eval_databases(from_organizer: DirectoryOrganizer, to_organizer: DirectoryOrganizer,
                        last_gen: Optional[Generation]=None):
    for f in os.listdir(from_organizer.eval_db_dir):
        benchmark_tag = tag_from_db_filename(f)

        if benchmark_tag is None:
            raise ValueError(f'Invalid eval db filename: {f} in {from_organizer.eval_db_dir}')

        if benchmark_tag == from_organizer.tag:
            logger.debug(f'Skipping eval db for self-eval: {f}')
            continue

        db = RatingDB(from_organizer.eval_db_filename(benchmark_tag))
        new_db = RatingDB(to_organizer.eval_db_filename(benchmark_tag))
        copy_eval_db(db, new_db, to_organizer.tag, last_gen)


def copy_eval_db(db: RatingDB, new_db: RatingDB, new_tag: str, last_gen: Optional[Generation]=None):
    db_id_map: Dict[int, int] = {}
    arena = Arena()
    for db_agent in db.fetch_agents():
        assert len(db_agent.roles) == 1

        if isinstance(db_agent.agent, MCTSAgent) and db_agent.roles == {AgentRole.TEST}:
            if last_gen is not None and db_agent.agent.gen > last_gen:
                continue
            agent = dataclasses.replace(db_agent.agent, tag=new_tag)
        else:
            agent = db_agent.agent

        iagent = arena.add_agent(agent, db_agent.roles, db=new_db)
        db_id_map[db_agent.db_id] = iagent.db_id

    for result in db.fetch_match_results():
        if result.agent_id1 not in db_id_map or result.agent_id2 not in db_id_map:
            continue
        new_db_id1 = db_id_map[result.agent_id1]
        new_db_id2 = db_id_map[result.agent_id2]
        new_db.commit_counts(new_db_id1, new_db_id2, result.counts, result.type)

    ratings_data = db.load_ratings(AgentRole.TEST)
    iagents = []
    ratings = []
    for data in ratings_data:
        if data.agent_id not in db_id_map:
            continue
        new_db_id = db_id_map[data.agent_id]
        iagent = arena.agent_lookup_db_id[new_db_id]
        iagents.append(iagent)
        ratings.append(data.rating)
    ratings = np.array(ratings)
    new_db.commit_ratings(iagents, ratings)

    benchmark_rating_data = db.load_ratings(AgentRole.BENCHMARK)
    iagents = []
    ratings = []
    committee = IndexSet()
    for i, data in enumerate(benchmark_rating_data):
        if data.agent_id not in db_id_map:
            continue
        new_db_id = db_id_map[data.agent_id]
        iagent = arena.agent_lookup_db_id[new_db_id]
        iagents.append(iagent)
        ratings.append(data.rating)
        committee.add(i)
    ratings = np.array(ratings)
    new_db.commit_ratings(iagents, ratings, committee)


def main():
    args = load_args()
    logging_params = LoggingParams.create(args)
    configure_logger(params=logging_params)

    if args.from_tag is None:
        raise ValueError('Required option: --from-tag/-f')
    if args.to_tag is None:
        raise ValueError('Required option: --to-tag/-t')
    if args.game is None:
        raise ValueError('Required option: --game/-g')

    assert game_index.is_valid_game_name(args.game), f'Invalid game name: {args.game}'

    from_params = RunParams(args.game, args.from_tag)
    to_params = RunParams(args.game, args.to_tag)

    from_organizer = DirectoryOrganizer(from_params, base_dir_root=Workspace)
    to_organizer = DirectoryOrganizer(to_params, base_dir_root=Workspace)

    if not os.path.isdir(from_organizer.base_dir):
        raise ValueError(f'From-directory does not exist: {from_organizer.base_dir}')
    if from_organizer.fork_info is not None:
        raise ValueError(f'From-directory is already a fork: {from_organizer.base_dir}')
    if os.path.isdir(to_organizer.base_dir):
        raise ValueError(f'To-directory already exists: {to_organizer.base_dir}')

    hard_fork = bool(args.hard_fork)
    retrain_models = bool(args.retrain_models)
    last_gen = args.last_gen

    if last_gen is not None:
        if last_gen <= 0:
            raise ValueError('last-self-play-gen must be greater than 0')

    to_organizer.dir_setup(paradigm=from_organizer.paradigm())

    if hard_fork:
        logger.info('Copying self-play data...')
        from_organizer.copy_self_play_data(to_organizer, last_gen)
    else:
        logger.info('Soft-linking self-play data...')
        from_organizer.soft_link_self_play_data(to_organizer, last_gen)

    if retrain_models:
        logger.info('Skipping model files...')
    elif hard_fork:
        logger.info('Copying model files...')
        from_organizer.copy_models_and_checkpoints(to_organizer, last_gen)
    else:
        logger.info('Soft-linking model files...')
        from_organizer.soft_link_models_and_checkpoints(to_organizer, last_gen)

    logger.info('Copying database files...')
    copy_databases(from_organizer, to_organizer, retrain_models, last_gen)

    logger.info('Copying binary files...')
    from_organizer.copy_binary(to_organizer)

    logger.info('Writing fork info...')
    to_organizer.write_fork_info(from_organizer, retrain_models, last_gen)

    logger.info('Fork complete!')


if __name__ == '__main__':
    main()
