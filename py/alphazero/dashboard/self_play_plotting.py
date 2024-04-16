"""
Used by launch_dashboard.py to create a self-play plot.
"""
from bokeh.plotting import figure

from typing import List


def create_self_play_figure(output_dir: str, game: str, tags: List[str]):
    return figure(title='TODO: show self-play metrics')
