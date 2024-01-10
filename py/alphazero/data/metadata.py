from dataclasses import dataclass
from natsort import natsorted
import os
from typing import Dict, Iterable, List

from alphazero.custom_types import Generation


class DoneFileInfo:
    """
    done.txt contains a key=value pair on each line. This class parses the file and stores the values.

    Note that the appropriate parsing of the values (e.g. to int or to float) is left to the caller.
    """
    def __init__(self, filename: str):
        self.mappings = {}
        with open(filename, 'r') as f:
            lines = list(f.readlines())

        for line in lines:
            line = line.strip()
            if line:
                tokens = line.split('=')
                assert len(tokens) == 2, line
                key = tokens[0].strip()
                value = tokens[1].strip()
                self.mappings[key] = value

    def __getitem__(self, key):
        return self.mappings[key]


@dataclass
class SelfPlayGameMetadata:
    """
    The metadata associated with a single self-play game. Each game is one-to-one with a file in
    the self-play-data directory.

    The files are produced by the c++ self-play process, which encodes a unique nanosecond timestamp
    and the number of positions in the game into the filename. This encoding allows us to perform
    sorting/organization of the files without having to parse the files themselves.
    """
    filename: str
    generation: int
    timestamp: int
    n_positions: int


@dataclass
class SelfPlayPositionMetadata:
    """
    The metadata associated with a single position in a self-play game. This is essentially a
    (self-play-filename, row-index) pair.
    """
    game_metadata: SelfPlayGameMetadata
    position_index: int


class GenerationMetadata:
    """
    The metadata associated with a generation-directory of self-play-game files. This is essentially
    a list of SelfPlayGameMetadata objects.

    Note that the self-play process might be writing to the generation-directory at the time this
    object is constructed. We thus need to be thoughtful about possible race-conditions:

    1. A single game-file might be in the middle of being written to.
    2. More game-files might be added to the directory during this object's lifetime.

    Race-condition 1 is addressed on the c++ side, by writing to a temporary file and then
    performing an atomic mv to the final filename.

    Race-condition 2 is addressed by the mechanics of how this class works. It only processes the
    game-files that exist at the time of construction. If further game-files are added to the
    directory, they are intentionally ignored.

    As a performance optimization, the AlphaZeroManager class, which manages the self-play
    process, writes a done.txt file to each generation-directory when all self-play games have
    been completed. The done.txt file contains summary statistics about the directory, like the
    total number of positions. This allows us to lazily load the game-metadata objects, if all we
    need is the summary statistics.
    """
    def __init__(self, full_gen_dir: str):
        self._loaded = False
        self.full_gen_dir = full_gen_dir
        self.generation = int(os.path.basename(full_gen_dir).split('-')[1])
        self._game_metadata_list = []

        done_file = os.path.join(full_gen_dir, 'done.txt')
        self.done = os.path.isfile(done_file)
        if self.done:
            done_file_info = DoneFileInfo(done_file)
            self.n_games = int(done_file_info['n_games'])
            self.n_positions = int(done_file_info['n_positions'])
            return

        self.n_positions = -1
        self.n_games = -1
        self.load()

    @property
    def game_metadata_list(self) -> List[SelfPlayGameMetadata]:
        """
        Returns a list of the SelfPlayGameMetadata's associated with the files in this directory,
        sorted by each file's unique timestamp.
        """
        self.load()
        return self._game_metadata_list

    def load(self):
        if self._loaded:
            return

        self._loaded = True
        for filename in os.listdir(self.full_gen_dir):
            if filename.startswith('.') or filename.endswith('.txt'):
                continue

            # 1685860410604914-10.ptd
            full_filename = os.path.join(self.full_gen_dir, filename)
            info = filename.split('.')[0].split('-')
            timestamp = int(info[0])
            n_positions = int(info[1])
            game_metadata = SelfPlayGameMetadata(full_filename, self.generation, timestamp,
                                                 n_positions)
            self._game_metadata_list.append(game_metadata)

        self._game_metadata_list.sort(key=lambda g: g.timestamp)  # oldest to newest
        n_positions = sum(g.n_positions for g in self._game_metadata_list)
        n_games = len(self._game_metadata_list)

        assert self.n_positions in (-1, n_positions), (self.n_positions, n_positions)
        assert self.n_games in (-1, n_games), (self.n_games, n_games)

        self.n_positions = n_positions
        self.n_games = n_games


class SelfPlayMetadata:
    """
    The metadata associated with a top-level self-play-data directory. This directory contains
    subdirectories named gen-0, gen-1, gen-2, ... Each of these subdirectories contains
    self-play-data game files.

    This is essentially a list of GenerationMetadata objects, one per subdirectory. When flattened,
    it represents a master list of individual positions.

    Current usage demands a manual reresh() call after constructing the SelfPlayMetadata object.
    """

    def __init__(self, db_filename: str):
        self.db_filename = db_filename

        self.self_play_dir = self_play_dir
        self.metadata: Dict[Generation, GenerationMetadata] = {}
        self.n_total_positions = 0
        self.n_total_games = 0
        self.last_done_gen = -1

    def _reload_from(self, start_gen: int):
        gen = start_gen
        while True:
            gen_dir = os.path.join(self.self_play_dir, f'gen-{gen}')
            if not os.path.isdir(gen_dir):
                break
            metadata = GenerationMetadata(gen_dir)
            if metadata.done:
                self.last_done_gen = max(metadata.generation, self.last_done_gen)
            self.metadata[gen] = metadata
            self.n_total_positions += metadata.n_positions
            self.n_total_games += metadata.n_games
            gen += 1

        if gen in self.metadata:
            # rm partially written generation
            metadata = self.metadata[gen]
            self.n_total_positions -= metadata.n_positions
            self.n_total_games -= metadata.n_games
            del self.metadata[gen]

    def refresh(self):
        self._reload_from(self.last_done_gen + 1)

    def get_window(self, start: int, stop: int) -> Iterable[SelfPlayPositionMetadata]:
        """
        Returns an iterable of SelfPlayPositionMetadata objects. The returned objects logically
        correspond to the slice master_list[start:stop], where master_list represents the list that
        would be obtained by flattening the self-play-data directory into individual positions.
        """
        i = 0
        for gen in sorted(self.metadata):
            if i >= stop:
                return

            gen_metadata = self.metadata[gen]
            n = gen_metadata.n_positions
            if i + n <= start:
                i += n
                continue

            for game_metadata in gen_metadata.game_metadata_list:
                for p in range(game_metadata.n_positions):
                    if i >= stop:
                        return
                    if i >= start:
                        yield SelfPlayPositionMetadata(game_metadata, p)
                    i += 1
