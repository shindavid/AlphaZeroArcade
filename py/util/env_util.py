import os


def get_output_dir():
    env_var = 'A0A_OUTPUT_DIR'
    x = os.environ.get(env_var, None)
    if x is None:
        raise Exception(f'Environment variable {env_var} is not set. '
                        f'Please run "source env_setup.sh" from repo root.')
    return x
