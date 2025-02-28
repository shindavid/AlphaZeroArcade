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


SELF_PLAY_TABLE_CREATE_CMDS = [
    """CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER,
            gen INTEGER,
            report_timestamp INTEGER,
            cache_hits INTEGER,
            cache_misses INTEGER,
            positions_evaluated INTEGER,
            batches_evaluated INTEGER,
            full_batches_evaluated INTEGER
            )""",

    """CREATE TABLE games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER,
            gen INTEGER,
            start_timestamp INTEGER,
            end_timestamp INTEGER,
            augmented_positions INTEGER,
            cumulative_augmented_positions INTEGER
            )""",

    """CREATE TABLE self_play_metadata (
            gen INTEGER PRIMARY KEY,
            positions_evaluated INTEGER DEFAULT 0,
            batches_evaluated INTEGER DEFAULT 0,
            games INTEGER DEFAULT 0,
            runtime INTEGER DEFAULT 0,
            augmented_positions INTEGER DEFAULT 0
            )""",

    """CREATE TRIGGER update_games AFTER INSERT ON games
            BEGIN
                UPDATE games
                SET cumulative_augmented_positions = CASE
                WHEN NEW.id = 1 THEN NEW.augmented_positions
                ELSE (SELECT cumulative_augmented_positions FROM games WHERE id = NEW.id - 1)
                        + NEW.augmented_positions
                END
                WHERE id = NEW.id;
            END""",
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
            subtype TEXT
            )""",

    """CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id1 INT,
            agent_id2 INT,
            agent1_wins INT,
            agent2_wins INT,
            draws INT
            )""",

    """CREATE UNIQUE INDEX IF NOT EXISTS lookup ON matches (agent_id1, agent_id2)""",

    """CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id INT UNIQUE,
            rating FLOAT,
            is_committee INT
            )""",

    """CREATE UNIQUE INDEX IF NOT EXISTS lookup ON ratings (agent_id)""",
]
