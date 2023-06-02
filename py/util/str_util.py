"""
Collection of str util functions.
"""


def center_text(text: str, n: int) -> str:
    """
    center_text('abc', 5) -> ' abc '
    center_text('abc', 6) -> ' abc  '
    center_text('abc', 7) -> '  abc  '
    """
    m = n - len(text)
    assert m >= 0
    a = m // 2
    b = m - a
    return (' ' * a) + text + (' ' * b)


def inject_arg(cmdline: str, arg_name: str, arg_value: str):
    """
    Takes a cmdline and adds {arg_name}={arg_value} into it, overriding any existing value for {arg_name}.
    """
    assert arg_name.startswith('-'), arg_name
    tokens = cmdline.split()
    for i, token in enumerate(tokens):
        if token == arg_name:
            tokens[i+1] = arg_value
            return ' '.join(tokens)

        if token.startswith(f'{arg_name}='):
            tokens[i] = f'{arg_name}={arg_value}'
            return ' '.join(tokens)

    return f'{cmdline} {arg_name} {arg_value}'


def inject_args(cmdline: str, kwargs: dict):
    """
    Takes a cmd-line string and injects the given key/value pairs into the string. The key/value pairs should be
    of the form {'--arg_name': 'arg_value', ...}.

    If --arg_name is already present in cmdline, then it will be replaced with --arg_name=arg_value.

    TODO:
    - support standalone args
    - support args with aliases like -v/--verbose
    """
    for arg_name, arg_value in kwargs.items():
        cmdline = inject_arg(cmdline, arg_name, arg_value)
    return cmdline
