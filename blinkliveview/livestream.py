"""Resilient livestream wrapper for handling unstable Blink streams."""

import asyncio
import logging
import ssl
import urllib.parse

_LOGGER = logging.getLogger(__name__)


class ResilientLiveStream:
    """Wrapper around BlinkLiveStream that handles connection issues more gracefully."""

    def __init__(self, camera, response):
        """Initialize ResilientLiveStream."""
        self.camera = camera
        self.command_id = response["command_id"]
        self.polling_interval = response["polling_interval"]
        self.target = urllib.parse.urlparse(response["server"])
        self.server = None
        self.clients = []
        self.target_reader = None
        self.target_writer = None
        self._stop_requested = False

    def add_auth_header_string_field(self, auth_header, field_string, max_length):
        """Add string field to authentication header."""
        field_bytes = field_string.encode("utf-8")[:max_length]
        field_bytes = field_bytes.ljust(max_length, b"\x00")
        field_length = len(field_bytes).to_bytes(4, byteorder="big")
        auth_header.extend(field_length)
        auth_header.extend(field_bytes)

    def get_auth_header(self):
        """Get authentication header."""
        auth_header = bytearray()
        serial_max_length = 16
        token_field_max_length = 64
        conn_id_max_length = 16

        magic_number = [0x00, 0x00, 0x00, 0x28]
        auth_header.extend(magic_number)

        serial = self.camera.serial
        self.add_auth_header_string_field(auth_header, serial, serial_max_length)

        client_id = urllib.parse.parse_qs(self.target.query).get("client_id", [0])[0]
        client_id_field = int(client_id).to_bytes(4, byteorder="big")
        auth_header.extend(client_id_field)

        static_field = [0x01, 0x08]
        auth_header.extend(static_field)

        token_length = token_field_max_length.to_bytes(4, byteorder="big")
        auth_header.extend(token_length)
        auth_header.extend([0x00] * token_field_max_length)

        conn_id = self.target.path.split("/")[-1].split("__")[0]
        self.add_auth_header_string_field(auth_header, conn_id, conn_id_max_length)

        trailer_static = [0x00, 0x00, 0x00, 0x01]
        auth_header.extend(trailer_static)

        return auth_header

    async def start(self, host="127.0.0.1", port=None):
        """Start the stream."""
        self.server = await asyncio.start_server(self.join, host, port)
        return self.server

    @property
    def socket(self):
        """Return the socket."""
        return self.server.sockets[0]

    @property
    def url(self):
        """Return the URL of the stream."""
        sockname = self.socket.getsockname()
        return f"tcp://{sockname[0]}:{sockname[1]}"

    @property
    def is_serving(self):
        """Check if the stream is active."""
        return self.server and self.server.is_serving()

    async def feed(self):
        """Connect to and stream from the target server."""
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        self.target_reader, self.target_writer = await asyncio.open_connection(
            self.target.hostname, self.target.port, ssl=ssl_context
        )

        auth_header = self.get_auth_header()
        self.target_writer.write(auth_header)
        await self.target_writer.drain()

        try:
            await asyncio.gather(self.recv(), self.send(), self.poll())
        except Exception:
            _LOGGER.exception("Error while handling stream")
        finally:
            _LOGGER.debug("Streaming was aborted, stopping server")
            self.stop()

    async def join(self, client_reader, client_writer):
        """Join client to the stream."""
        self.clients.append(client_writer)

        try:
            while not client_writer.is_closing():
                data = await client_reader.read(1024)
                if not data:
                    _LOGGER.debug("Client disconnected")
                    break
                await asyncio.sleep(0)
        except ConnectionResetError:
            _LOGGER.debug("Client connection reset")
        except Exception:
            _LOGGER.exception("Error while handling client")
        finally:
            self.clients.remove(client_writer)
            if not client_writer.is_closing():
                client_writer.close()

            if not self.clients:
                _LOGGER.debug("Last client disconnected, stopping server")
                self.stop()

    async def recv(self):
        """Copy data from target to clients - with resilience for incomplete packets."""
        consecutive_errors = 0
        max_consecutive_errors = 10

        try:
            _LOGGER.debug("Starting resilient copy from target to clients")
            while not self.target_reader.at_eof() and not self._stop_requested:
                try:
                    # Read header from the target server
                    data = await asyncio.wait_for(
                        self.target_reader.read(9), timeout=10.0
                    )

                    if len(data) < 9:
                        if len(data) == 0:
                            _LOGGER.debug("No data received, stream may have ended")
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                _LOGGER.warning(
                                    "Too many consecutive errors, ending stream"
                                )
                                break
                            await asyncio.sleep(0.1)
                            continue
                        _LOGGER.debug(
                            "Insufficient header data: %d bytes, expected 9 - skipping",
                            len(data),
                        )
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            break
                        continue

                    # Reset error counter on successful header read
                    consecutive_errors = 0

                    msgtype = data[0]
                    sequence = int.from_bytes(data[1:5], byteorder="big")
                    payload_length = int.from_bytes(data[5:9], byteorder="big")

                    if payload_length <= 0:
                        _LOGGER.debug("Invalid payload length: %d", payload_length)
                        continue

                    if payload_length > 65535:
                        _LOGGER.debug(
                            "Payload length too large: %d, skipping", payload_length
                        )
                        continue

                    # Read payload - use readexactly with timeout, but handle partial reads
                    try:
                        payload_data = await asyncio.wait_for(
                            self._read_payload(payload_length), timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        _LOGGER.debug("Timeout reading payload, continuing")
                        continue

                    if payload_data is None or len(payload_data) < payload_length:
                        # Instead of breaking, just skip this packet and continue
                        _LOGGER.debug(
                            "Incomplete payload: got %d bytes, expected %d - skipping packet",
                            len(payload_data) if payload_data else 0,
                            payload_length,
                        )
                        continue

                    # Skip packets other than msgtype 0x00 (regular video stream)
                    if msgtype != 0x00:
                        _LOGGER.debug("Skipping unsupported msgtype %d", msgtype)
                        continue

                    # Skip video payloads missing 0x47 (transport stream packet start)
                    if payload_data[0] != 0x47:
                        _LOGGER.debug("Skipping video payload missing 0x47 at start")
                        continue

                    # Send data to all connected clients
                    for writer in self.clients:
                        if not writer.is_closing():
                            try:
                                writer.write(payload_data)
                                await writer.drain()
                            except Exception:
                                _LOGGER.debug("Error writing to client, skipping")

                    await asyncio.sleep(0)

                except asyncio.TimeoutError:
                    _LOGGER.debug("Timeout waiting for data, continuing")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        break
                    continue

        except ssl.SSLError as e:
            if e.reason != "APPLICATION_DATA_AFTER_CLOSE_NOTIFY":
                _LOGGER.exception("SSL error while receiving data")
        except Exception:
            _LOGGER.exception("Error while receiving data")
        finally:
            if self.target_writer and not self.target_writer.is_closing():
                self.target_writer.close()
            _LOGGER.debug("Receiving ended")

    async def _read_payload(self, length):
        """Read payload with handling for incomplete data."""
        data = bytearray()
        remaining = length

        while remaining > 0:
            chunk = await self.target_reader.read(min(remaining, 8192))
            if not chunk:
                break
            data.extend(chunk)
            remaining -= len(chunk)

        return bytes(data) if len(data) == length else None

    async def send(self):
        """Send keep-alive and latency-stats messages to the server."""
        latency_stats_packet = [
            0x12,
            0x00,
            0x00,
            0x03,
            0xE8,
            0x00,
            0x00,
            0x00,
            0x18,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
        ]
        every10s = 0
        sequence = 0
        try:
            while (
                not self._stop_requested
                and self.target_writer
                and not self.target_writer.is_closing()
            ):
                if (every10s % 10) == 0:
                    every10s = 0
                    sequence += 1
                    sequence_bytes = sequence.to_bytes(4, byteorder="big")

                    keepalive_packet = [
                        0x0A,
                        *sequence_bytes,
                        0x00,
                        0x00,
                        0x00,
                        0x00,
                    ]

                    _LOGGER.debug("Sending keep-alive packet")
                    self.target_writer.write(bytearray(keepalive_packet))
                    await self.target_writer.drain()

                _LOGGER.debug("Sending latency-stats packet")
                self.target_writer.write(bytearray(latency_stats_packet))
                await self.target_writer.drain()

                every10s += 1
                await asyncio.sleep(1)
        except Exception:
            _LOGGER.exception("Error while sending keep-alive or latency-stats")
        finally:
            if self.target_reader:
                self.target_reader.feed_eof()
            _LOGGER.debug("Sending ended")

    async def poll(self):
        """Poll the command API for the stream."""
        from blinkpy import api

        try:
            while (
                not self._stop_requested
                and self.target_reader
                and not self.target_reader.at_eof()
            ):
                _LOGGER.debug("Polling command API")
                response = await api.request_command_status(
                    self.camera.sync.blink, self.camera.network_id, self.command_id
                )
                _LOGGER.debug("Polling response: %s", response)

                if response.get("status_code", 0) != 908:
                    _LOGGER.error("Polling command API failed: %s", response)
                    break

                for commands in response.get("commands", []):
                    if commands.get("id") == self.command_id:
                        state_condition = commands.get("state_condition")
                        if state_condition not in ("new", "running"):
                            return

                await asyncio.sleep(self.polling_interval)
        except Exception:
            _LOGGER.exception("Error while polling command API")
        finally:
            _LOGGER.debug("Done polling command API")
            try:
                from blinkpy import api

                await api.request_command_done(
                    self.camera.sync.blink, self.camera.network_id, self.command_id
                )
            except Exception:
                pass

    def stop(self):
        """Stop the stream."""
        self._stop_requested = True
        _LOGGER.debug("Stopping server, closing remaining connections")
        if self.server and self.server.is_serving():
            _LOGGER.debug("Closing listen server")
            self.server.close()
        if self.target_writer and not self.target_writer.is_closing():
            _LOGGER.debug("Closing target writer")
            self.target_writer.close()
        for writer in self.clients:
            if not writer.is_closing():
                _LOGGER.debug("Closing client writer")
                writer.close()
        _LOGGER.debug("All remaining connections closed")


async def init_resilient_livestream(camera):
    """Initialize a resilient livestream for a camera."""
    from blinkpy import api

    response = await api.request_camera_liveview(
        camera.sync.blink,
        camera.sync.network_id,
        camera.camera_id,
        camera_type=camera.camera_type,
    )
    if not response["server"].startswith("immis://"):
        raise NotImplementedError("Unsupported: {}".format(response["server"]))
    return ResilientLiveStream(camera, response)
