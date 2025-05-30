DEFAULT_LOOP_CONTROLLER_PORT = 1111
LOCALHOST_IP = '127.0.0.1'
DEFAULT_REMOTE_PLAY_PORT = 1234


CLIENTS_TABLE_CREATE_CMDS = [
    """CREATE TABLE clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT,
            port INTEGER,
            role TEXT,
            start_timestamp INTEGER,
            cuda_device TEXT
            )""",
]


# Should match what is emitted by core::PerfStat::to_json() in c++
PERF_STATS_COLUMNS = [
    # SearchThreadPerfStats
    'cache_hits',
    'cache_misses',

    'wait_for_game_slot_time_ns',
    'cache_mutex_acquire_time_ns',
    'cache_insert_time_ns',
    'batch_prepare_time_ns',
    'batch_write_time_ns',
    'wait_for_nn_eval_time_ns',
    'mcts_time_ns',

    # NNEvalLoopPerfStats
    'positions_evaluated',
    'batches_evaluated',
    'full_batches_evaluated',

    'wait_for_search_threads_time_ns',
    'pipeline_wait_time_ns',
    'pipeline_schedule_time_ns',

    'batch_datas_allocated',

    # LoopControllerPerfStats
    'pause_time_ns',
    'model_load_time_ns',
    'total_time_ns',
]

SELF_PLAY_TABLE_CREATE_CMDS = [
    # metrics columns should match json keys for PerfStats in c++
    """CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER,
            gen INTEGER,
            report_timestamp INTEGER,
            %s
            )""" % ', '.join(
        '%s INTEGER' % col for col in PERF_STATS_COLUMNS
    ),

    """CREATE TABLE self_play_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gen INTEGER,
            positions INTEGER,
            cumulative_positions INTEGER,
            positions_evaluated INTEGER DEFAULT 0,
            batches_evaluated INTEGER DEFAULT 0,
            games INTEGER DEFAULT 0,
            file_size INTEGER DEFAULT 0
            )""",
]


TRAINING_TABLE_CREATE_CMDS = [
    """CREATE TABLE training (
            gen INTEGER PRIMARY KEY,
            minibatch_size INTEGER,
            n_minibatches INTEGER,
            training_start_ts INTEGER,
            training_end_ts INTEGER DEFAULT 0,
            window_start INTEGER,
            window_end INTEGER,
            window_sample_rate FLOAT
            )""",

    """CREATE TABLE training_heads (
            gen INTEGER,
            head_name TEXT,
            loss FLOAT,
            loss_weight FLOAT
            )""",

    """CREATE INDEX training_heads_idx ON training_heads (gen)""",
]


RATINGS_TABLE_CREATE_CMDS = [
    """CREATE TABLE IF NOT EXISTS matches (
            tag TEXT,
            mcts_gen INT,
            ref_strength INT,
            mcts_wins INT,
            draws INT,
            ref_wins INT
            )""",

    """CREATE UNIQUE INDEX IF NOT EXISTS lookup ON matches (tag, mcts_gen, ref_strength)""",

    """CREATE TABLE IF NOT EXISTS ratings (
            tag TEXT,
            mcts_gen INT,
            n_games INT,
            rating FLOAT
            )""",

    """CREATE UNIQUE INDEX IF NOT EXISTS lookup ON ratings (tag, mcts_gen)""",
]


ARENA_TABLE_CREATE_CMDS = [
    """CREATE TABLE IF NOT EXISTS mcts_agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gen INT,
            n_iters INT,
            tag TEXT,
            is_zero_temp INT
            )""",

    """CREATE TABLE IF NOT EXISTS ref_agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type_str TEXT,
            strength_param TEXT,
            strength INT,
            tag TEXT
            )""",

    """CREATE TABLE IF NOT EXISTS agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sub_id INT,
            subtype TEXT,
            role TEXT
            )""",

    """CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id1 INT,
            agent_id2 INT,
            agent1_wins INT,
            agent2_wins INT,
            draws INT,
            type TEXT
            )""",

    """CREATE UNIQUE INDEX IF NOT EXISTS lookup ON matches (agent_id1, agent_id2)""",

    """CREATE TABLE IF NOT EXISTS benchmark_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id INT UNIQUE,
            rating FLOAT,
            is_committee INT
            )""",

    """CREATE UNIQUE INDEX IF NOT EXISTS lookup ON benchmark_ratings (agent_id)""",

    """CREATE TABLE IF NOT EXISTS evaluator_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id INT UNIQUE,
            rating FLOAT
            )""",

    """CREATE UNIQUE INDEX IF NOT EXISTS lookup ON evaluator_ratings (agent_id)""",
]
