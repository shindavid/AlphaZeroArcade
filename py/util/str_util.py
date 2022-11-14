
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
