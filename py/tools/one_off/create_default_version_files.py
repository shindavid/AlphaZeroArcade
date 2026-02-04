#!/usr/bin/env python3
"""
This script creates version files in all existing output directories that lack them.
It does NOT know which paradigm was used to generated the directory, so it defaults to 'alpha0'.
If the paradigm is known and is different, please manually edit the created version files.
"""
from alphazero.logic.run_params import RunParams
from alphazero.servers.loop_control.base_dir import Workspace
from alphazero.servers.loop_control.directory_organizer import DirectoryOrganizer

from pathlib import Path


def main():
    output_dir = Path(Workspace.output_dir())

    for game_dir in output_dir.iterdir():
        if not game_dir.is_dir():
            continue

        for dir in game_dir.iterdir():
            if not dir.is_dir():
                continue
            organizer = DirectoryOrganizer(RunParams(game=game_dir.name, tag=dir.name), Workspace)
            version_file = Path(organizer.version_filename)
            if not version_file.exists():
                organizer.write_version_file('alpha0')

if __name__ == '__main__':
    main()
