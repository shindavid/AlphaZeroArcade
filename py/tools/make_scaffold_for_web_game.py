#! /usr/bin/env python3

"""
This script is used to add scaffolding for a new web game.

It adds some files into web/games/<game_name>.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys


ESLINT_CONFIG = '''// Auto-generated via: LAUNCHCMD

import baseConfig from '../shared/eslint.config.base.js';
export default baseConfig;
'''

INDEX_HTML = '''<!-- Auto-generated via: LAUNCHCMD -->

<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/png" href="/a0a.png" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>GAMEUPPER</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
'''

PACKAGE_JSON = '''{
  "__generated_by": "LAUNCHCMD",
  "name": "GAMELOWER",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "lint": "eslint .",
    "preview": "vite preview"
  }
}
'''

MAIN_JSX = '''// Auto-generated via: LAUNCHCMD

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../../shared/styles/shared.css'
import GAMEUPPERApp from './GAMEUPPER.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <GAMEUPPERApp />
  </StrictMode>,
)
'''

GAME_JSX = '''// Auto-generated via: LAUNCHCMD

import './GAMEUPPER.css';
import '../../shared/shared.css';
import { GameAppBase } from '../../shared/GameAppBase';

export default class GAMEUPPERApp extends GameAppBase {
  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      board: null,
    };
  }

  renderBoard() {
    if (!this.state.board) return null;
    // TODO: FILL IN THE BOARD RENDERING LOGIC HERE
  }
}
'''

GAME_CSS = '''/* Auto-generated via: LAUNCHCMD */

/* TODO: add css elements for GAMEUPPER */
'''

VITE_CONFIG_JS = '''// Auto-generated via: LAUNCHCMD

import baseConfig from '../shared/vite.config.base.js';
export default baseConfig;
'''

def good_print(msg):
    print(f"✅ {msg}")


def bad_print(msg):
    print(f"❌ {msg}")


class Scaffolder:
    def __init__(self, game_camel_case : str, overwrite: bool):
        self.overwrite = overwrite
        self.cmd = ' '.join(shlex.quote(arg) for arg in sys.argv)

        self.game_lower = game_camel_case.lower()
        self.game_camel_case = game_camel_case

        self.game_dir = f'web/games/{self.game_lower}'
        self.public_dir = f'{self.game_dir}/public'
        self.src_dir = f'{self.game_dir}/src'

        self.eslint_config_path = f'{self.game_dir}/eslint.config.js'
        self.index_html_path = f'{self.game_dir}/index.html'
        self.package_json_path = f'{self.game_dir}/package.json'
        self.main_jsx_path = f'{self.src_dir}/main.jsx'
        self.game_jsx_path = f'{self.src_dir}/{self.game_camel_case}.jsx'
        self.game_css_path = f'{self.src_dir}/{self.game_camel_case}.css'
        self.vite_config_js_path = f'{self.game_dir}/vite.config.js'

    def _format(self, template: str, escape_json=False) -> str:
        x = template
        cmd = self.cmd
        if escape_json:
            cmd = json.dumps(cmd)[1:-1]
        replacements = {
            'GAMELOWER': self.game_lower,
            'GAMEUPPER': self.game_camel_case,
            'LAUNCHCMD': cmd,
        }
        for a, b in replacements.items():
            x = x.replace(a, b)

        return x

    def _make_dir(self, path: str):
        if os.path.isdir(path):
            good_print(f"Skipping directory creation (already exists): {path}")
            return

        if os.path.exists(path):
            bad_print(f"Path exists but is not a directory: {path}")
            sys.exit(1)

        os.makedirs(path, exist_ok=True)
        good_print(f"Created directory: {path}")

    def _write_file(self, path: str, content: str):
        if os.path.isfile(path):
            if not self.overwrite:
                good_print(f"Skipping file creation (already exists): {path}")
                return

            prev_content = open(path, 'r').read()
            if prev_content == content:
                good_print(f"Skipping file creation (content is unchanged): {path}")
                return

            with open(path, 'w') as f:
                f.write(content)

            good_print(f"Overwrote existing file: {path}")
            return

        with open(path, 'w') as f:
            f.write(content)
        good_print(f"Created new file: {path}")

    def make_game_dir(self):
        self._make_dir(self.game_dir)

    def write_eslint_config(self):
        self._write_file(self.eslint_config_path, self._format(ESLINT_CONFIG))

    def write_index_html(self):
        self._write_file(self.index_html_path, self._format(INDEX_HTML))

    def write_package_json(self):
        self._write_file(self.package_json_path, self._format(PACKAGE_JSON, escape_json=True))

    def make_public_dir(self):
        self._make_dir(self.public_dir)

    def add_a0a_png_soft_link(self):
        src = '../../../public/a0a.png'
        dst = f'{self.public_dir}/a0a.png'
        if os.path.islink(dst) and os.readlink(dst) == src:
            good_print(f"Skipping symlink creation (already exists): {dst}")
            return
        cmd = f'ln -sf ../../../public/a0a.png {self.public_dir}/a0a.png'
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        good_print(f"Created symlink: {self.public_dir}/a0a.png")

    def make_src_dir(self):
        self._make_dir(self.src_dir)

    def write_main_jsx(self):
        self._write_file(self.main_jsx_path, self._format(MAIN_JSX))

    def write_game_jsx(self):
        self._write_file(self.game_jsx_path, self._format(GAME_JSX))

    def write_game_css(self):
        self._write_file(self.game_css_path, self._format(GAME_CSS))

    def write_vite_config_js(self):
        self._write_file(self.vite_config_js_path, self._format(VITE_CONFIG_JS))

def get_args():
    parser = argparse.ArgumentParser(description='Add scaffolding for a new web game.')
    parser.add_argument('-g', '--game', help='Name of the game to scaffold, in CamelCase')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Overwrite existing files if they already exist')
    return parser.parse_args()


def main():
    args = get_args()
    game = args.game

    if not game:
        print("Please provide a game name (-g/--game).")
        return

    if game == game.lower():
        print("Game name should be in CamelCase (e.g., TicTacToe).")
        return


    # chdir to the repo root:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    assert os.path.isfile(os.path.join(repo_root, 'REPO_ROOT_MARKER'))
    os.chdir(repo_root)

    s = Scaffolder(game, bool(args.overwrite))

    s.make_game_dir()
    s.write_eslint_config()
    s.write_index_html()
    s.write_package_json()
    s.make_public_dir()
    s.add_a0a_png_soft_link()
    s.make_src_dir()
    s.write_main_jsx()
    s.write_game_jsx()
    s.write_game_css()
    s.write_vite_config_js()


if __name__ == '__main__':
    main()
