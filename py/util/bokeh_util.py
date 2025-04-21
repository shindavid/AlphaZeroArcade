from bokeh.models import ColumnDataSource, CustomJSTickFormatter, Plot
from bokeh.palettes import Category10, Category20
import numpy as np

from typing import List, Optional


def get_colors(n: int) -> List[str]:
    if n <= 2:
        return Category10[3][:n]
    if n <= 10:
        return Category10[n]
    if n <= 20:
        return Category20[n]

    raise ValueError(f"Too many colors requested: {n}")


def make_time_tick_formatter(resolution: str):
    """
    resolution should be 'm' or 's' (minutes or seconds)
    """
    if resolution == 'm':
        code = """
                    var total_seconds = tick;
                    var days = Math.floor(total_seconds / 86400);
                    var hours = Math.floor((total_seconds % 86400) / 3600);
                    var minutes = Math.floor((total_seconds % 3600) / 60);
                    if (days > 0) {
                        return days + "d " + hours + "h " + minutes + "m";
                    } else {
                        return hours + "h " + minutes + "m";
                    }
                """
    elif resolution == 's':
        code = """
                    var total_seconds = tick;
                    var days = Math.floor(total_seconds / 86400);
                    var hours = Math.floor((total_seconds % 86400) / 3600);
                    var minutes = Math.floor((total_seconds % 3600) / 60);
                    var seconds = total_seconds % 60;
                    if (days > 0) {
                        return days + "d " + hours + "h " + minutes + "m " + seconds + "s";
                    } else if (hours > 0) {
                        return hours + "h " + minutes + "m " + seconds + "s";
                    } else {
                        return minutes + "m " + seconds + "s";
                    }
                """
    else:
        raise ValueError(f"Invalid resolution: {resolution}")
    return CustomJSTickFormatter(code=code)


def add_stacked_area(p: Plot, source: ColumnDataSource, y_cols: List[str], x_col: str='x',
                     colors: Optional[List[str]]=None, legend_labels: Optional[List[str]]=None):
    n = len(y_cols)
    if colors is None:
        from bokeh.palettes import Viridis256
        colors = Viridis256[:n]

    if legend_labels is None:
        legend_labels = y_cols

    # prev = np.zeros(len(source.data[x_col]))
    prev = 0
    x = source.data[x_col]

    for i, y_col in enumerate(y_cols):
        top = np.array(source.data[y_col]) + prev
        p.varea(x=x, y1=prev, y2=top, fill_color=colors[i], alpha=0.8,
                legend_label=legend_labels[i])
        prev = top
