import argparse
import hashlib
import inspect
import os
import shutil
import subprocess
import tempfile
from typing import List, Union


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def is_valid_path_component(path_component):
    try:
        # Attempt to join the path component with a dummy path
        # If it's a valid component, it should not raise an exception
        joined_path = os.path.join('dummy_path', path_component)
        return os.path.split(joined_path)[1] == path_component
    except (ValueError, OSError):
        return False


_sha256_cache = {}


def sha256sum(filename, use_cache=True):
    """
    Returns the sha256 checksum of the file specified by filename.

    If use_cache is True, then as an optimization, if the checksum has already been computed and the file has not been
    modified since the checksum was computed, then the cached checksum is returned.
    """
    if not use_cache:
        return sha256sum_helper(filename)

    mtime = os.path.getmtime(filename)
    if filename in _sha256_cache:
        cached_mtime, cached_checksum = _sha256_cache[filename]
        if cached_mtime == mtime:
            return cached_checksum

    checksum = sha256sum_helper(filename)

    _sha256_cache[filename] = (mtime, checksum)
    return checksum


def sha256sum_helper(filename):
    # https://stackoverflow.com/a/44873382/543913
    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def tar_and_remotely_copy(src_dir, dst_tar):
    """
    Tar up `src_dir` and copy the tar file to `dst_tar`. Optimizes for the case that `dst_tar` is
    on a network filesystem.
    """
    # Generate a unique temporary tar file path
    fd, local_tar = tempfile.mkstemp(suffix=".tar", prefix="tarcopy_", dir="/tmp")
    os.close(fd)  # Close the open file descriptor

    try:
        subprocess.run(["tar", "-cf", local_tar, "-C", os.path.dirname(src_dir),
                        os.path.basename(src_dir)], check=True)
        shutil.copy2(local_tar, dst_tar)
    finally:
        if os.path.exists(local_tar):
            os.remove(local_tar)


def untar_remote_file_to_local_directory(src_tar, dst_dir):
    """
    Extracts the tar archive `src_tar` into `dst_dir`. Optimizes for the case that `src_tar` is on a
    network filesystem.
    """
    fd, local_tar = tempfile.mkstemp(suffix=".tar", prefix="untar_", dir="/tmp")
    os.close(fd)
    shutil.copy2(src_tar, local_tar)

    try:
        subprocess.run(["tar", "-xf", local_tar, "-C", dst_dir], check=True)
    finally:
        if os.path.exists(local_tar):
            os.remove(local_tar)


def make_hidden_filename(filename):
    """
    Returns a filename formed by prepending a '.' to the filename part of filename.
    """
    head, tail = os.path.split(filename)
    return os.path.join(head, '.' + tail)


def atomic_cp(src, dst, intermediate=None):
    """
    Equivalent to the unix cmd:

    cp src dst

    The above cmd is not atomic, however. This function is.

    It works by first copying src to a temporary file, then renaming the temporary file to dst. This
    works because the unix cmd "mv" is atomic (as long as the files are in the same filesystem).

    The location of the temporary file is specified by intermediate. If intermediate is None, then
    the location is created by prepending a '.' to the filename part of dst. It is the
    responsibility of the caller to ensure that:

    1. intermediate is on the same file system as dst
    2. intermediate does not already exist
    3. the temporary existance of intermediate does not cause any problems
    """
    if intermediate is None:
        intermediate = make_hidden_filename(dst)

    assert not os.path.exists(intermediate), intermediate
    shutil.copyfile(src, intermediate)
    os.rename(intermediate, dst)


def atomic_softlink(target, link_name, intermediate=None):
    """
    Equivalent to the unix cmd:

    ln -sf target link_name

    The above cmd is not atomic, however. This function is.

    It works by first creating a temporary symlink intermediate, then renaming the temporary
    symlink to link_name. This works because the unix cmd "mv" is atomic (as long as the files
    are in the same filesystem).

    The location of the temporary file is specified by intermediate. If intermediate is None, then
    the location is created by prepending a '.' to the filename part of link_name. It is the
    responsibility of the caller to ensure that:

    1. intermediate is on the same file system as link_name
    2. intermediate does not already exist
    3. the temporary existance of intermediate does not cause any problems
    """
    if intermediate is None:
        intermediate = make_hidden_filename(link_name)

    assert not os.path.exists(intermediate), intermediate
    os.symlink(target, intermediate)
    os.rename(intermediate, link_name)


def get_function_arguments(ignore: Union[str, List[str], None]=None):
    """
    If called from within a function, returns a dict mapping the names of the function's arguments
    to their values.

    If ignore is specified, then the returned dict will not contain any keys in ignore. The type
    of ignore should be a list of strings; if it is a string, then it will be interpreted as a
    singleton list of that string.

    Example:

    class Foo:
        def __init__(self, a=1, b=2):
            print(get_function_arguments('self'))

    foo = Foo(b=3)  # will print {'a': 1, 'b': 3}

    Courtesy of ChatGPT.
    """
    ignore = [] if ignore is None else ignore
    if type(ignore) is str:
        # interpret string arg as a singleton list
        ignore = [ignore]

    # Get the frame of the caller
    frame = inspect.currentframe().f_back

    # Extract the function's argument names
    arg_info = inspect.getargvalues(frame)
    arg_names = arg_info.args

    # Build a dictionary of the function's arguments
    args = {name: arg_info.locals[name] for name in arg_names if name not in ignore}

    # Don't forget to delete the frame object to avoid reference cycles
    # (The Python docs caution that frame objects may cause memory leaks if not cleaned up properly)
    del frame

    return args


def find_largest_gap(items: list):
    """
    Given a list of at least 2 items, returns the consecutive pair of items that have the largest
    gap between them.
    """
    assert len(items) >= 2
    largest_gap = None
    largest_gap_pair = None
    for i in range(len(items) - 1):
        gap = items[i + 1] - items[i]
        if largest_gap is None or gap > largest_gap:
            largest_gap = gap
            largest_gap_pair = (items[i], items[i + 1])
    return largest_gap_pair


class CustomHelpFormatter(argparse.HelpFormatter):
    """
    [dshin] I don't like argparse's default help formatting, so I wrote this custom formatter.

    Default format:

        -t TAG, --tag TAG     tag for this run (e.g. "v1")

    Custom format:
        -t/--tag TAG          tag for this run (e.g. "v1")
    """
    def _format_action_invocation(self, action):
        if action.option_strings:
            return '/'.join(action.option_strings) + ' ' + self._format_args(action, action.dest.upper())
        else:
            return super()._format_action_invocation(action)


def create_symlink(src: str, dst: str):
    """
    Creates a symbolic link from `src` to `dst`.
    - If `src` is a file, creates a symlink to the file.
    - If `src` is a directory, creates symlinks for all files inside `dst`.

    Args:
        src (str): Source file or directory path.
        dst (str): Destination path where the symlink(s) should be created.
    """
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source path does not exist: {src}")

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.isfile(src):
        # If src is a file, create a symlink directly
        os.symlink(src, dst)
        print(f"Created symlink: {dst} -> {src}")

    elif os.path.isdir(src):
        # If src is a directory, create a directory at dst (if it doesn’t exist)
        os.makedirs(dst, exist_ok=True)

        # Iterate over all items in the source directory
        for item in os.listdir(src):
            src_item = os.path.join(src, item)
            dst_item = os.path.join(dst, item)

            # Create a symlink for each file/directory inside src
            os.symlink(src_item, dst_item)
            print(f"Created symlink: {dst_item} -> {src_item}")

    else:
        raise ValueError(f"Invalid source type: {src} (Not a file or directory)")


def copy_file_to_folder(src: str, dst_folder: str):
    """
    Copies a file from `src` to `dst_folder`, preserving metadata.

    Args:
        src (str): Path to the source file.
        dst_folder (str): Path to the destination folder.

    Raises:
        FileNotFoundError: If `src` does not exist.
        ValueError: If `src` is not a file.
    """
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source file does not exist: {src}")
    if not os.path.isfile(src):
        raise ValueError(f"Source is not a file: {src}")

    # Ensure the destination folder exists
    os.makedirs(dst_folder, exist_ok=True)

    # Copy file to the destination folder (preserving metadata)
    dst_path = os.path.join(dst_folder, os.path.basename(src))
    shutil.copy2(src, dst_path)

    print(f"Copied: {src} -> {dst_path}")


