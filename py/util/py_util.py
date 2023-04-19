import datetime
import hashlib
import os
import shutil


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


def timed_print(s, *args, **kwargs):
    t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f'{t} {s}', *args, **kwargs)


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

    It works by first copying src to a temporary file, then renaming the temporary file to dst. This works
    because the unix cmd "mv" is atomic (as long as the file systems are on the same machine).

    The location of the temporary file is specified by intermediate. If intermediate is None, then the location is
    created by prepending a '.' to the filename part of dst. It is the responsibility of the caller to ensure that:

    1. intermediate is on the same file system as dst
    2. intermediate does not already exist
    3. the temporary existance of intermediate does not cause any problems
    """
    if intermediate is None:
        intermediate = make_hidden_filename(dst)

    assert not os.path.exists(intermediate), intermediate
    shutil.copyfile(src, intermediate)
    os.rename(intermediate, dst)
