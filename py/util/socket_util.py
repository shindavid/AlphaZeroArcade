import socket

from typing import Optional


def recvall(sock: socket.socket, n: int, timeout: Optional[float]=None) -> bytearray:
    """Receive exactly n bytes from the socket, looping internally if necessary.

    Args:
    - sock: the socket object to read from. Must be in blocking mode
    - n: the number of bytes to read
    - timeout: the timeout in seconds, or None for blocking mode (default)

    Raises:
    - ValueError if n <= 0, or if socket is in non-blocking mode, or if timeout is not positive
    - socket.timeout if timeout is specified and the socket operation times out.
    - ConnectionError if the socket was gracefully closed by peer.
    - ConnectionResetError if the socket was forcefully reset by peer.
    - OSError if there was an OS-related error.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if sock.gettimeout() is not None:
        raise ValueError("Socket must be in blocking mode")

    if timeout is not None:
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        sock.settimeout(timeout)

    data = bytearray()
    try:
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("Socket gracefully closed by peer")
            data.extend(packet)
    finally:
        sock.settimeout(None)

    return data
