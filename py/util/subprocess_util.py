import logging
import os
import signal
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Union


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


def wait_for(proc: Union[subprocess.Popen, List[subprocess.Popen]], timeout=None,
             expected_return_code: Optional[int] = 0, print_fn=print):
    """
    Waits until proc is complete, validates returncode, and returns stdout.

    If proc is a list, waits for all of them, validates all their return codes, and returns a list
    of their stdouts.
    """
    if not isinstance(proc, list):
        procs = [proc]
    else:
        procs = proc

    stdouts = []
    error_info = []
    for p in procs:
        assert isinstance(p, subprocess.Popen), f'Unexpected type p={type(p)} (proc={type(proc)})'

        stdout, stderr = p.communicate(timeout=timeout)
        stdouts.append(stdout)
        if expected_return_code not in (p.returncode, None):
            print_fn(f'Expected rc={expected_return_code}, got rc={p.returncode}')
            print_fn('----------------------------')
            print_fn('STDERR:')
            print_fn(stderr)
            print_fn('----------------------------')
            error_info.append((p.returncode, p.args))

    if error_info:
        # raise an exception for the first process that failed
        raise subprocess.CalledProcessError(*error_info[0])

    if len(stdouts) == 1:
        return stdouts[0]
    return stdouts


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


def terminate_processes(procs: Iterable[subprocess.Popen], timeout: float = 5.0):
    procs = list(procs)
    for proc in procs:
        if proc.poll() is None:
            logger.debug(f'Terminating process {proc.pid}')
            try:
                proc.terminate()
            except Exception as e:
                logger.error(f'Error terminating process {proc.pid}: {e}')
        else:
            logger.debug(f'Process {proc.pid} already exited with code {proc.returncode}')

    for proc in procs:
        if proc.poll() is None:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(f'Process {proc.pid} did not terminate in time; killing it')
                try:
                    proc.kill()
                    proc.wait(timeout=timeout)
                except Exception as e:
                    logger.error(f'Failed to kill process {proc.pid}: {e}')
            except Exception as e:
                logger.error(f'Error waiting for process {proc.pid}: {e}')
