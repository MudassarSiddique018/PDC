import asyncio
import socket
import struct

import numpy as np
import pytest

from minicluster.common import protocol as p


def test_socket_roundtrip_with_payload():
    a, b = socket.socketpair()
    with a, b:
        payload = p.pack_arrays(X=np.arange(12.0).reshape(3, 4), y=np.array([1, 0, 1]))
        p.send_message(a, {"type": "task", "task_id": "T1"}, payload)
        msg = p.recv_message(b)
    assert msg.type == "task" and msg.header["task_id"] == "T1"
    arrays = p.unpack_arrays(msg.payload)
    np.testing.assert_array_equal(arrays["X"], np.arange(12.0).reshape(3, 4))
    np.testing.assert_array_equal(arrays["y"], [1, 0, 1])


def test_many_messages_are_framed_independently():
    a, b = socket.socketpair()
    with a, b:
        for i in range(50):
            p.send_message(a, {"type": "heartbeat", "i": i})
        got = [p.recv_message(b).header["i"] for _ in range(50)]
    assert got == list(range(50))


def test_closed_peer_raises_connection_error():
    a, b = socket.socketpair()
    a.close()
    with b, pytest.raises(ConnectionError):
        p.recv_message(b)


def test_header_without_type_is_rejected():
    a, b = socket.socketpair()
    with a, b:
        raw = b'{"no_type": 1}'
        a.sendall(struct.pack("!IQ", len(raw), 0) + raw)
        with pytest.raises(p.ProtocolError):
            p.recv_message(b)


def test_oversized_frame_is_rejected():
    a, b = socket.socketpair()
    with a, b:
        a.sendall(struct.pack("!IQ", p.MAX_HEADER_BYTES + 1, 0))
        with pytest.raises(p.ProtocolError):
            p.recv_message(b)


def test_asyncio_and_blocking_sides_interoperate():
    async def main():
        received = {}

        async def handle(reader, writer):
            msg = await p.async_recv(reader)
            received["msg"] = msg
            await p.async_send(writer, {"type": "welcome", "worker_id": "PC-01"})
            writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        def client():
            with socket.create_connection(("127.0.0.1", port)) as s:
                p.send_message(s, {"type": "register", "name": "PC-01"}, b"\x00" * 1000)
                return p.recv_message(s)

        reply = await asyncio.to_thread(client)
        server.close()
        await server.wait_closed()
        return received["msg"], reply

    msg, reply = asyncio.run(main())
    assert msg.type == "register" and len(msg.payload) == 1000
    assert reply.header["worker_id"] == "PC-01"
