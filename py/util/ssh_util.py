import os
import fcntl


def get_pub_key(key_type='ed25519'):
    """
    Returns the public key of the current user.
    """
    try:
        with open(os.path.expanduser(f'~/.ssh/id_{key_type}.pub')) as f:
            return f.read().strip()
    except FileNotFoundError:
        raise Exception(f'Could not find ssh public key of type {key_type}')


def add_to_authorized_keys(pub_key):
    _add_to_ssh_file(pub_key, 'authorized_keys')


def _add_to_ssh_file(pub_key, filename):
    """
    Adds the given public key to the current user's ~/.ssh/{filename} file,
    ensuring correct permissions, concurrency safety, and no duplicates.

    Courtesy of ChatGPT-o1.
    """
    auth_keys_path = os.path.expanduser(f'~/.ssh/{filename}')
    auth_keys_dir = os.path.dirname(auth_keys_path)

    # 1) Ensure ~/.ssh directory exists (700 is typical)
    if not os.path.isdir(auth_keys_dir):
        os.makedirs(auth_keys_dir, mode=0o700, exist_ok=True)
    else:
        # Fix directory perms if you want to be thorough:
        ssh_dir_mode = os.stat(auth_keys_dir).st_mode & 0o777
        if ssh_dir_mode != 0o700:
            os.chmod(auth_keys_dir, 0o700)

    # 2) Ensure file exists with 600 perms
    #    If it doesn't exist, create it explicitly with 0o600.
    if not os.path.exists(auth_keys_path):
        # Create the file with the correct perms
        fd = os.open(auth_keys_path, os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(fd)
    else:
        # If it does exist, fix perms if needed
        auth_keys_mode = os.stat(auth_keys_path).st_mode & 0o777
        if auth_keys_mode != 0o600:
            os.chmod(auth_keys_path, 0o600)

    # 3) Open the file in a+ mode (append+read), acquire an exclusive lock,
    #    check for the key, and append if needed.
    with open(auth_keys_path, 'a+', encoding='utf-8') as f:
        # Acquire exclusive lock (blocks until lock is acquired)
        fcntl.flock(f, fcntl.LOCK_EX)

        # Seek back to the beginning so we can read the current contents
        f.seek(0)
        contents = f.read()

        # Only append if the key isn't already present
        if pub_key not in contents:
            f.write(pub_key.rstrip('\n') + '\n')
            f.flush()
            os.fsync(f.fileno())
