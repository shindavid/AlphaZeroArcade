from bokeh.palettes import Category10, Category20

from typing import List


def get_colors(n: int) -> List[str]:
    if n <= 2:
        return Category10[3][:n]
    if n <= 10:
        return Category10[n]
    if n <= 20:
        return Category20[n]

    raise ValueError(f"Too many colors requested: {n}")
