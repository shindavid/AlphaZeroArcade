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
import re

def parse_for_paradigm(log_file: Path) -> str:
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            match = re.search(r'--type\s+([^\s-]+)', content)
        if match:
            return match.group(1)
        return None

    except Exception as e:
        print(f"Error reading {log_file}: {e}")

def create_version_file(organizer):
    version_file = Path(organizer.version_filename)
    if not version_file.exists():
        log_dir = Path(organizer.logs_dir) / 'self-play-server'
        for entry in log_dir.iterdir():
            if not entry.is_file():
                continue

            paradigm = parse_for_paradigm(entry)
            if paradigm:
                organizer.write_version_file(paradigm)
                print(f"Created version file for {organizer.game}/{organizer.tag} with paradigm {paradigm}")
                return

        raise ValueError(f"Could not determine paradigm for {organizer.game}/{organizer.tag}.")

def main():
    output_dir = Path(Workspace.output_dir())

    for game_dir in output_dir.iterdir():
        if not game_dir.is_dir():
            continue

        for dir in game_dir.iterdir():
            if not dir.is_dir():
                continue
            organizer = DirectoryOrganizer(RunParams(game=game_dir.name, tag=dir.name), Workspace)
            create_version_file(organizer)

if __name__ == '__main__':
    main()
