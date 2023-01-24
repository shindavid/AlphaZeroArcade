import subprocess
from typing import Union, List


def Popen(cmd: Union[str, List[str]], **kwargs) -> subprocess.Popen:
    """
    This is a convenience wrapper around subprocess.Popen, with different defaults:

    * shell: either True/False depending on whether cmd is a str or not
    * stdout/stdin/stderr: subprocess.PIPE
    * encoding: 'utf-8'
    """
    kwargs = dict(**kwargs)
    kwargs['shell'] = kwargs.get('shell', isinstance(cmd, str))
    kwargs['stdout'] = kwargs.get('stdout', subprocess.PIPE)
    kwargs['stdin'] = kwargs.get('stdin', subprocess.PIPE)
    kwargs['stderr'] = kwargs.get('stderr', subprocess.PIPE)
    kwargs['encoding'] = kwargs.get('encoding', 'utf-8')

    return subprocess.Popen(cmd, **kwargs)
