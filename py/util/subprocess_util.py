import subprocess
from typing import Union, List, Dict, Any


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
    """
    return subprocess.Popen(cmd, **defaultize_kwargs(cmd, **kwargs))


def wait_for(proc: subprocess.Popen):
    """
    Waits until proc is complete, validates returncode, and return.
    """
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        print(stdout)
        print(stderr)
        raise subprocess.CalledProcessError(proc.returncode, proc.args)


def run(cmd: Union[str, List[str]], validate_rc=True, **kwargs) -> subprocess.CompletedProcess:
    """
    Convenience wrapper around subprocess.run(), using the defaults of defaultize_kwargs().
    """
    proc = subprocess.run(cmd, **defaultize_kwargs(cmd, **kwargs))
    if validate_rc and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
    return proc
