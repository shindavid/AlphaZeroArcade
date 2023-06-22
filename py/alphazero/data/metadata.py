import os
from typing import Dict, List

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


class SelfPlayGameMetadata:
    def __init__(self, filename: str):
        self.filename = filename
        info = os.path.split(filename)[1].split('.')[0].split('-')  # 1685860410604914-10.ptd
        self.timestamp = int(info[0])
        self.n_positions = int(info[1])


class SelfPlayPositionMetadata:
    def __init__(self, game_metadata: SelfPlayGameMetadata, position_index: int):
        self.game_metadata = game_metadata
        self.position_index = position_index


class GenerationMetadata:
    def __init__(self, full_gen_dir: str):
        self._loaded = False
        self.full_gen_dir = full_gen_dir
        self._game_metadata_list = []

        done_file = os.path.join(full_gen_dir, 'done.txt')
        if os.path.isfile(done_file):
            done_file_info = DoneFileInfo(done_file)
            self.n_games = int(done_file_info['n_games'])
            self.n_positions = int(done_file_info['n_positions'])
            return

        self.n_positions = 0
        self.n_games = 0
        self.load()

    @property
    def game_metadata_list(self):
        self.load()
        return self._game_metadata_list

    def load(self):
        if self._loaded:
            return

        self._loaded = True
        for filename in os.listdir(self.full_gen_dir):
            if filename.startswith('.') or filename.endswith('.txt'):
                continue
            full_filename = os.path.join(self.full_gen_dir, filename)
            game_metadata = SelfPlayGameMetadata(full_filename)
            self._game_metadata_list.append(game_metadata)

        self._game_metadata_list.sort(key=lambda g: -g.timestamp)  # newest to oldest
        self.n_positions = sum(g.n_positions for g in self._game_metadata_list)
        self.n_games = len(self._game_metadata_list)


class SelfPlayMetadata:
    def __init__(self, self_play_dir: str, first_gen=0):
        self.self_play_dir = self_play_dir
        self.metadata: Dict[Generation, GenerationMetadata] = {}
        self.n_total_positions = 0
        self.n_total_games = 0
        for gen_dir in os.listdir(self_play_dir):
            assert gen_dir.startswith('gen-'), gen_dir
            generation = int(gen_dir.split('-')[1])
            if generation < first_gen:
                continue
            full_gen_dir = os.path.join(self_play_dir, gen_dir)
            metadata = GenerationMetadata(full_gen_dir)
            self.metadata[generation] = metadata
            self.n_total_positions += metadata.n_positions
            self.n_total_games += metadata.n_games

    def get_window(self, n_window: int) -> List[SelfPlayPositionMetadata]:
        window = []
        cumulative_n_positions = 0
        for generation in reversed(sorted(self.metadata.keys())):  # newest to oldest
            gen_metadata = self.metadata[generation]
            n = len(gen_metadata.game_metadata_list)
            i = 0
            while cumulative_n_positions < n_window and i < n:
                game_metadata = gen_metadata.game_metadata_list[i]
                cumulative_n_positions += game_metadata.n_positions
                i += 1
                for p in range(game_metadata.n_positions):
                    window.append(SelfPlayPositionMetadata(game_metadata, p))
        return window
