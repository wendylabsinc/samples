from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import socket


async def pipe_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    finally:
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()


async def handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    remote_host: str,
    remote_port: int,
) -> None:
    try:
        remote_reader, remote_writer = await open_remote_connection(remote_host, remote_port)
    except Exception:
        client_writer.close()
        await client_writer.wait_closed()
        return

    await asyncio.gather(
        pipe_stream(client_reader, remote_writer),
        pipe_stream(remote_reader, client_writer),
        return_exceptions=True,
    )


async def open_remote_connection(
    remote_host: str,
    remote_port: int,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    infos = socket.getaddrinfo(remote_host, remote_port, type=socket.SOCK_STREAM)
    infos.sort(key=lambda item: 0 if item[0] == socket.AF_INET else 1)

    last_error: Exception | None = None
    for family, _, _, _, sockaddr in infos:
        host = sockaddr[0]
        port = sockaddr[1]
        try:
            return await asyncio.open_connection(host=host, port=port, family=family)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not connect to {remote_host}:{remote_port}") from last_error


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, required=True)
    args = parser.parse_args()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop_event.set)

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, args.target_host, args.target_port),
        args.listen_host,
        args.listen_port,
    )

    async with server:
        await stop_event.wait()


if __name__ == "__main__":
    asyncio.run(main())
