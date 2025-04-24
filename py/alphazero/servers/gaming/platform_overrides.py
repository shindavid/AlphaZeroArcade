import os

from typing import Any, Dict

_platform = None

def get_platform():
    global _platform
    if _platform is None:
        _platform = os.getenv('PLATFORM', 'default')
    return _platform

def update_cpp_bin_args(args: Dict[str, Any]):
    """
    On RunPod, we set the number of game threads to 8 by default.

    This is not a great solution, as it makes assumptions about the pod being used on RunPod that
    may not be true. However, it is a quick fix to get things working.

    In the future, we should do a dynamic on-the-fly auto-configuration to determine the optimal
    number of threads (as well as the optimal batch size and number of game-slots).
    """
    platform = get_platform()
    if platform == 'runpod':
        keys = ['--num-game-threads', '-t']
        overrode = False
        for key in keys:
            if key in args:
                args[key] = 8
                overrode = True
                break

        if not overrode:
            args['--num-game-threads'] = 8
