import logging
import os
import signal
import subprocess
import threading
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

    If proc is a list, waits until for all of them, validates all their return codes, and returns a
    list of their stdouts. If any of the processes exit with an unexpected return code, raises an
    exception before waiting for the others to finish.
    """
    if isinstance(proc, subprocess.Popen):
        stdout, stderr = proc.communicate(timeout=timeout)
        if expected_return_code not in (proc.returncode, None):
            print_fn(f'Expected rc={expected_return_code}, got rc={proc.returncode}')
            print_fn('----------------------------')
            print_fn(f'args: {proc.args}')
            print_fn('----------------------------')
            print_fn(f'STDOUT:\n{stdout}')
            print_fn('----------------------------')
            print_fn(f'STDERR:\n{stderr}')
            print_fn('----------------------------')
            raise subprocess.CalledProcessError(proc.returncode, proc.args)
        return stdout

    procs = list(proc)
    cond = threading.Condition()
    timed_out_pids = []
    completed_procs = []
    failed_procs = []
    stdouts = {}
    stderrs = {}

    def run_proc(p: subprocess.Popen):
        try:
            stdout, sterr = p.communicate(timeout=timeout)
            stdouts[p.pid] = stdout
            stderrs[p.pid] = sterr
            if expected_return_code not in (p.returncode, None):
                failed_procs.append(p)
        except subprocess.TimeoutExpired:
            p.kill()
            timed_out_pids.append(p.pid)
            failed_procs.append(p)

        completed_procs.append(p)
        with cond:
            cond.notify_all()

    [threading.Thread(target=run_proc, args=(p,)).start() for p in proc]
    with cond:
        cond.wait_for(lambda: len(completed_procs) == len(procs) or failed_procs)

    timed_out_pids = list(timed_out_pids)
    failed_procs = list(failed_procs)

    for pid in timed_out_pids:
        logger.warning(f'Process {pid} timed out and was killed')

    for p in failed_procs:
        stdout = stdouts.get(p.pid, None)
        stderr = stderrs.get(p.pid, None)
        print_fn(f'Process {p.pid} failed with return code {p.returncode}')
        print_fn('----------------------------')
        print_fn(f'args: {p.args}')
        print_fn('----------------------------')
        print_fn(f'STDOUT:\n{stdout}')
        print_fn('----------------------------')
        print_fn(f'STDERR:\n{stderr}')
        print_fn('----------------------------')

    if failed_procs:
        raise Exception('One or more processes failed. See above.')

    assert len(stdouts) == len(procs)
    return [stdouts[p.pid] for p in procs]


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
