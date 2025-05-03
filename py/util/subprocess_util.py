import logging
import os
import signal
import subprocess
from typing import Union, List, Dict, Any, Optional


logger = logging.getLogger(__name__)


def defaultize_kwargs(cmd: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
    """
    Takes kwargs and adds the following mappings if the corresponding keys are not present:

    * shell: either True/False depending on whether cmd is a str or not
    * stderr: subprocess.PIPE
    * stdin: subprocess.PIPE
    * stdout: subprocess.PIPE
    * encoding: 'utf-8'

    Returns the corresponding dict.
    """
    kwargs = dict(**kwargs)
    kwargs['shell'] = kwargs.get('shell', isinstance(cmd, str))
    kwargs['stdout'] = kwargs.get('stdout', subprocess.PIPE)
    kwargs['stdin'] = kwargs.get('stdin', subprocess.PIPE)
    kwargs['stderr'] = kwargs.get('stderr', subprocess.PIPE)
    kwargs['encoding'] = kwargs.get('encoding', 'utf-8')
    return kwargs


def Popen(cmd: Union[str, List[str]], **kwargs) -> subprocess.Popen:
    """
    Convenience wrapper around subprocess.Popen(), using the defaults of defaultize_kwargs().

    Also, if cmd is a str, it prepends "exec" in front of it. This is a nice hack to make it so that proc.kill()
    kills the process. See: https://stackoverflow.com/a/13143013/543913
    """
    if isinstance(cmd, str):
        cmd = 'exec ' + cmd
    return subprocess.Popen(cmd, **defaultize_kwargs(cmd, **kwargs))


def wait_for(proc: subprocess.Popen, timeout=None, expected_return_code: Optional[int] = 0,
             print_fn=print):
    """
    Waits until proc is complete, validates returncode, and returns stdout.
    """
    stdout, stderr = proc.communicate(timeout=timeout)
    if expected_return_code not in (proc.returncode, None):
        print_fn(f'Expected rc={expected_return_code}, got rc={proc.returncode}')
        print_fn('----------------------------')
        print_fn('STDERR:')
        print_fn(stderr)
        print_fn('----------------------------')
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
    return stdout


def run(cmd: Union[str, List[str]], validate_rc=True, **kwargs) -> subprocess.CompletedProcess:
    """
    Convenience wrapper around subprocess.run(), using the defaults of defaultize_kwargs().

    Also, if cmd is a str, it prepends "exec" in front of it. This is a nice hack to make it so that proc.kill()
    kills the process. See: https://stackoverflow.com/a/13143013/543913
    """
    if isinstance(cmd, str):
        cmd = 'exec ' + cmd
    proc = subprocess.run(cmd, **defaultize_kwargs(cmd, **kwargs))
    if validate_rc and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
    return proc


def safe_killpg(pid, signal: signal.Signals):
    try:
        logger.debug(f'Killing process group {pid} with signal {signal}')
        os.killpg(pid, signal)
    except ProcessLookupError:
        pass  # process group already gone
