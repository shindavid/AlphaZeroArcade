"""
At the top level of the repo checkout, users can maintain a config.txt file, to store arbitrary key-value pairs, in
the following format:

----------------------------------
# config.txt example
key1 = value1
key2=  value2  # some comment

 key3 =value3
----------------------------------

This allows for customization of things like output file locations without committing those customizations into the
repository.

The Config class defined here provides an API to access those key-value pairs.
"""
import argparse
import os

from util.repo_util import Repo


DEFAULT_FILENAME = os.path.join(Repo.root(), 'config.txt')


def decomment(line: str) -> str:
    """
    Strips pound-comments. TODO: do this better
    """
    pound = line.rfind('#')
    if pound != -1:
        return line[:pound]
    return line


class Config:
    _instance = None

    def __init__(self, filename: str = DEFAULT_FILENAME):
        self.filename = filename
        self._dict = {}
        if not os.path.isfile(filename):
            return

        with open(filename, 'r') as f:
            for orig_line in f:
                line = decomment(orig_line).strip()
                if not line:
                    continue
                eq = line.find('=')
                assert eq != -1, orig_line
                key = line[:eq].strip()
                value = line[eq+1:].strip()
                assert key not in self._dict, key
                self._dict[key] = value

    def get(self, key: str, default_value=None):
        return self._dict.get(key, default_value)

    def add_parser_argument(self, key: str, parser: argparse.ArgumentParser, *args, **kwargs):
        """
        Invokes parser.add_argument(*args, **kwargs), after first...

        - Adding default=self.get(key) to kwargs
        - Appending to kwargs['help'] info about the default value and where it came from
        """
        filename = os.path.relpath(self.filename, Repo.root())
        kwargs = dict(**kwargs)
        assert 'default' not in kwargs
        assert 'help' in kwargs
        help = kwargs['help']
        value = self._dict.get(key, None)
        if value is not None:
            kwargs['default'] = value
            kwargs['help'] = f'{help} (default: {value} [{filename}:{key}])'
        else:
            kwargs['help'] = f'{help} [{filename}:{key}]'
        if help == argparse.SUPPRESS:
            kwargs['help'] = help

        parser.add_argument(*args, **kwargs)

    @staticmethod
    def instance() -> 'Config':
        if Config._instance is None:
            Config._instance = Config()
        return Config._instance
