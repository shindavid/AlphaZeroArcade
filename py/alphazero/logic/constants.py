DEFAULT_LOOP_CONTROLLER_PORT = 1111
LOCALHOST_IP = '127.0.0.1'


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
            augmented_positions INTEGER DEFAULT 0
            )""",

    """CREATE TABLE timestamps (
            gen INTEGER,
            client_id INTEGER,
            start_timestamp INTEGER DEFAULT 0,
            end_timestamp INTEGER DEFAULT 0,
            PRIMARY KEY (gen, client_id)
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
            loss_weight FLOAT,
            accuracy FLOAT
            )""",

    """CREATE INDEX training_heads_idx ON training_heads (gen)""",
]


RATINGS_TABLE_CREATE_CMDS = [
    """CREATE TABLE IF NOT EXISTS matches (
            mcts_gen INT,
            mcts_iters INT,
            ref_strength INT,
            mcts_wins INT,
            draws INT,
            ref_wins INT
            )""",

    """CREATE UNIQUE INDEX IF NOT EXISTS lookup ON matches (mcts_gen, mcts_iters, ref_strength)""",

    """CREATE TABLE IF NOT EXISTS ratings (
            mcts_gen INT,
            mcts_iters INT,
            n_games INT,
            rating FLOAT
            )""",

    """CREATE UNIQUE INDEX IF NOT EXISTS lookup ON ratings (mcts_gen, mcts_iters)""",
]
