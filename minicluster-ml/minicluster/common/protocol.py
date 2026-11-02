"""Wire protocol shared by the coordinator and the workers.

Every message on the TCP socket is a single frame:

    +----------------+-----------------+----------------+-----------------+
    | header_len (4) | payload_len (8) | header (JSON)  | payload (bytes) |
    +----------------+-----------------+----------------+-----------------+

* ``header_len`` / ``payload_len`` are big-endian unsigned integers.
* ``header`` is a UTF-8 JSON object that always contains a ``type`` field.
* ``payload`` is optional binary data (NumPy arrays packed with ``np.savez``)
  so large datasets never have to be JSON-encoded.

Both a blocking (socket) and an asyncio (StreamReader/StreamWriter)
implementation are provided, because the worker uses classic threads while
the coordinator is built on asyncio.
"""

from __future__ import annotations

import asyncio
import io
import json
import socket
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1
_PREFIX = struct.Struct("!IQ")
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024 * 1024


class MsgType:
    """Message types exchanged between coordinator and worker."""

    # worker -> coordinator
    REGISTER = "register"
    HEARTBEAT = "heartbeat"
    RESULT = "result"
    # coordinator -> worker
    WELCOME = "welcome"
    REJECT = "reject"
    TASK = "task"
    SHUTDOWN = "shutdown"


class ProtocolError(Exception):
    """Raised when a peer sends a malformed frame."""


@dataclass
class Message:
    header: dict[str, Any]
    payload: bytes = field(default=b"")

    @property
    def type(self) -> str:
        return self.header.get("type", "")


def encode(header: dict[str, Any], payload: bytes = b"") -> bytes:
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _PREFIX.pack(len(raw), len(payload)) + raw + payload


def _decode_prefix(prefix: bytes) -> tuple[int, int]:
    header_len, payload_len = _PREFIX.unpack(prefix)
    if header_len > MAX_HEADER_BYTES or payload_len > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"frame too large (header={header_len}, payload={payload_len})")
    return header_len, payload_len


def _decode_header(raw: bytes) -> dict[str, Any]:
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid header: {exc}") from exc
    if not isinstance(header, dict) or "type" not in header:
        raise ProtocolError("header must be a JSON object with a 'type' field")
    return header


# --------------------------------------------------------------------------- #
# Blocking sockets (worker side)
# --------------------------------------------------------------------------- #
def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        chunk = sock.recv_into(view[got:], n - got)
        if chunk == 0:
            raise ConnectionError("peer closed the connection")
        got += chunk
    return bytes(buf)


def send_message(sock: socket.socket, header: dict[str, Any], payload: bytes = b"") -> None:
    sock.sendall(encode(header, payload))


def recv_message(sock: socket.socket) -> Message:
    header_len, payload_len = _decode_prefix(_recv_exact(sock, _PREFIX.size))
    header = _decode_header(_recv_exact(sock, header_len))
    payload = _recv_exact(sock, payload_len) if payload_len else b""
    return Message(header, payload)


# --------------------------------------------------------------------------- #
# asyncio streams (coordinator side)
# --------------------------------------------------------------------------- #
async def async_send(writer: asyncio.StreamWriter, header: dict[str, Any], payload: bytes = b"") -> None:
    writer.write(encode(header, payload))
    await writer.drain()


async def async_recv(reader: asyncio.StreamReader) -> Message:
    try:
        prefix = await reader.readexactly(_PREFIX.size)
        header_len, payload_len = _decode_prefix(prefix)
        header = _decode_header(await reader.readexactly(header_len))
        payload = await reader.readexactly(payload_len) if payload_len else b""
    except asyncio.IncompleteReadError as exc:
        raise ConnectionError("peer closed the connection") from exc
    return Message(header, payload)


# --------------------------------------------------------------------------- #
# NumPy array packing
# --------------------------------------------------------------------------- #
def pack_arrays(**arrays: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def unpack_arrays(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
