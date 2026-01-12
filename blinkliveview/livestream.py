"""Resilient livestream wrapper for handling unstable Blink streams."""

import asyncio
import collections
import logging
import ssl
import time
import urllib.parse
from typing import Optional

_LOGGER = logging.getLogger(__name__)

# Configuration constants for stream optimization
BUFFER_SIZE = 50  # Number of packets to buffer for jitter smoothing
BUFFER_LOW_WATERMARK = 10  # Start sending when buffer reaches this level
SOCKET_BUFFER_SIZE = 262144  # 256KB socket buffer for better throughput
MAX_CONSECUTIVE_ERRORS = 15  # More tolerance before giving up
RECONNECT_MAX_ATTEMPTS = 5  # Maximum reconnection attempts
RECONNECT_BASE_DELAY = 1.0  # Base delay for exponential backoff
RECONNECT_MAX_DELAY = 30.0  # Maximum delay between reconnection attempts
HEALTH_CHECK_INTERVAL = 5.0  # Seconds between health checks
PACKET_TIMEOUT = 15.0  # Timeout for receiving packets (increased from 10)
PAYLOAD_TIMEOUT = 8.0  # Timeout for reading payload (increased from 5)
SLOW_CLIENT_THRESHOLD = 100  # Drop packets if client buffer exceeds this
KEEPALIVE_INTERVAL = 1.0  # Send keepalive every second
KEEPALIVE_FULL_INTERVAL = 10  # Full keepalive packet every N seconds


class PacketBuffer:
    """Thread-safe packet buffer for smoothing network jitter."""

    def __init__(self, max_size: int = BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=max_size)
        self.max_size = max_size
        self._lock = asyncio.Lock()
        self._data_available = asyncio.Event()

    async def put(self, packet: bytes) -> None:
        """Add a packet to the buffer."""
        async with self._lock:
            self.buffer.append(packet)
            self._data_available.set()

    async def get(self) -> Optional[bytes]:
        """Get a packet from the buffer."""
        async with self._lock:
            if self.buffer:
                packet = self.buffer.popleft()
                if not self.buffer:
                    self._data_available.clear()
                return packet
            return None

    async def get_batch(self, max_count: int = 10) -> list[bytes]:
        """Get multiple packets from the buffer for batch processing."""
        async with self._lock:
            packets = []
            for _ in range(min(max_count, len(self.buffer))):
                packets.append(self.buffer.popleft())
            if not self.buffer:
                self._data_available.clear()
            return packets

    async def wait_for_data(self, timeout: float = 1.0) -> bool:
        """Wait for data to be available in the buffer."""
        try:
            await asyncio.wait_for(self._data_available.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    @property
    def size(self) -> int:
        """Current buffer size."""
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        """Check if buffer has enough data to start streaming."""
        return len(self.buffer) >= BUFFER_LOW_WATERMARK

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self._data_available.clear()


class ConnectionHealth:
    """Monitor connection health and detect issues proactively."""

    def __init__(self):
        self.last_packet_time: float = time.monotonic()
        self.packets_received: int = 0
        self.packets_dropped: int = 0
        self.errors: int = 0
        self.reconnects: int = 0
        self._healthy = True

    def packet_received(self) -> None:
        """Record a successful packet reception."""
        self.last_packet_time = time.monotonic()
        self.packets_received += 1
        self._healthy = True

    def packet_dropped(self) -> None:
        """Record a dropped packet."""
        self.packets_dropped += 1

    def error_occurred(self) -> None:
        """Record an error."""
        self.errors += 1

    def reconnect_occurred(self) -> None:
        """Record a reconnection."""
        self.reconnects += 1

    @property
    def seconds_since_last_packet(self) -> float:
        """Seconds since last packet was received."""
        return time.monotonic() - self.last_packet_time

    @property
    def is_healthy(self) -> bool:
        """Check if connection appears healthy."""
        # Consider unhealthy if no packets for too long
        return self._healthy and self.seconds_since_last_packet < PACKET_TIMEOUT

    @property
    def drop_rate(self) -> float:
        """Calculate packet drop rate."""
        total = self.packets_received + self.packets_dropped
        if total == 0:
            return 0.0
        return self.packets_dropped / total

    def reset(self) -> None:
        """Reset health metrics for new connection."""
        self.last_packet_time = time.monotonic()
        self.errors = 0
        self._healthy = True


class ResilientLiveStream:
    """Wrapper around BlinkLiveStream that handles connection issues more gracefully."""

    def __init__(self, camera, response):
        """Initialize ResilientLiveStream."""
        self.camera = camera
        self.command_id = response["command_id"]
        self.polling_interval = response["polling_interval"]
        self.target = urllib.parse.urlparse(response["server"])
        self.server = None
        self.clients: list = []
        self.target_reader: Optional[asyncio.StreamReader] = None
        self.target_writer: Optional[asyncio.StreamWriter] = None
        self._stop_requested = False

        # New optimization components
        self.packet_buffer = PacketBuffer()
        self.health = ConnectionHealth()
        self._reconnect_attempts = 0
        self._client_write_buffers: dict = {}  # Track per-client pending writes

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
        """Start the stream server with optimized socket settings."""
        # Create server with optimized socket options
        self.server = await asyncio.start_server(
            self.join,
            host,
            port,
            reuse_address=True,
            start_serving=True,
        )

        # Set socket buffer sizes for better throughput
        for sock in self.server.sockets:
            try:
                sock.setsockopt(
                    __import__("socket").SOL_SOCKET,
                    __import__("socket").SO_SNDBUF,
                    SOCKET_BUFFER_SIZE,
                )
                sock.setsockopt(
                    __import__("socket").SOL_SOCKET,
                    __import__("socket").SO_RCVBUF,
                    SOCKET_BUFFER_SIZE,
                )
            except Exception as e:
                _LOGGER.debug("Could not set socket buffer size: %s", e)

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
        """Connect to and stream from the target server with auto-reconnection."""
        while not self._stop_requested:
            try:
                await self._connect_and_stream()
            except asyncio.CancelledError:
                _LOGGER.debug("Feed cancelled")
                break
            except Exception as e:
                _LOGGER.warning("Stream connection failed: %s", e)
                self.health.error_occurred()

            if self._stop_requested:
                break

            # Attempt reconnection with exponential backoff
            if self._reconnect_attempts < RECONNECT_MAX_ATTEMPTS:
                delay = min(
                    RECONNECT_BASE_DELAY * (2**self._reconnect_attempts),
                    RECONNECT_MAX_DELAY,
                )
                _LOGGER.info(
                    "Reconnecting in %.1fs (attempt %d/%d)",
                    delay,
                    self._reconnect_attempts + 1,
                    RECONNECT_MAX_ATTEMPTS,
                )
                self._reconnect_attempts += 1
                self.health.reconnect_occurred()
                await asyncio.sleep(delay)
            else:
                _LOGGER.error(
                    "Max reconnection attempts (%d) reached, stopping",
                    RECONNECT_MAX_ATTEMPTS,
                )
                break

        self.stop()

    async def _connect_and_stream(self):
        """Establish connection and start streaming."""
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        _LOGGER.debug("Connecting to %s:%s", self.target.hostname, self.target.port)

        # Connect with timeout
        self.target_reader, self.target_writer = await asyncio.wait_for(
            asyncio.open_connection(
                self.target.hostname, self.target.port, ssl=ssl_context
            ),
            timeout=10.0,
        )

        # Set socket options on the connection
        sock = self.target_writer.get_extra_info("socket")
        if sock:
            try:
                sock.setsockopt(
                    __import__("socket").SOL_SOCKET,
                    __import__("socket").SO_RCVBUF,
                    SOCKET_BUFFER_SIZE,
                )
                # Enable TCP keepalive
                sock.setsockopt(
                    __import__("socket").SOL_SOCKET,
                    __import__("socket").SO_KEEPALIVE,
                    1,
                )
            except Exception as e:
                _LOGGER.debug("Could not set socket options: %s", e)

        # Send authentication
        auth_header = self.get_auth_header()
        self.target_writer.write(auth_header)
        await self.target_writer.drain()

        # Reset health and reconnect counter on successful connection
        self.health.reset()
        self._reconnect_attempts = 0
        self.packet_buffer.clear()

        _LOGGER.debug("Connected and authenticated, starting stream tasks")

        try:
            await asyncio.gather(
                self.recv(),
                self._distribute_packets(),
                self.send(),
                self.poll(),
                self._health_monitor(),
            )
        finally:
            if self.target_writer and not self.target_writer.is_closing():
                self.target_writer.close()
                try:
                    await self.target_writer.wait_closed()
                except Exception:
                    pass

    async def _health_monitor(self):
        """Monitor connection health and trigger reconnection if needed."""
        while not self._stop_requested and self.target_reader:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

            if self._stop_requested:
                break

            if not self.health.is_healthy:
                _LOGGER.warning(
                    "Connection unhealthy (no packets for %.1fs), triggering reconnect",
                    self.health.seconds_since_last_packet,
                )
                # Close the connection to trigger reconnection
                if self.target_writer and not self.target_writer.is_closing():
                    self.target_writer.close()
                break

            # Log health stats periodically
            if (
                self.health.packets_received > 0
                and self.health.packets_received % 1000 == 0
            ):
                _LOGGER.debug(
                    "Health: %d received, %d dropped (%.2f%%), %d errors, %d reconnects",
                    self.health.packets_received,
                    self.health.packets_dropped,
                    self.health.drop_rate * 100,
                    self.health.errors,
                    self.health.reconnects,
                )

    async def join(self, client_reader, client_writer):
        """Join client to the stream with tracking for slow client handling."""
        self.clients.append(client_writer)
        self._client_write_buffers[id(client_writer)] = 0

        _LOGGER.debug("Client joined, total clients: %d", len(self.clients))

        try:
            while not client_writer.is_closing() and not self._stop_requested:
                # Just keep connection alive, data is pushed by _distribute_packets
                try:
                    data = await asyncio.wait_for(
                        client_reader.read(1024), timeout=30.0
                    )
                    if not data:
                        _LOGGER.debug("Client disconnected (no data)")
                        break
                except asyncio.TimeoutError:
                    # Client is still connected, just no data received (normal)
                    continue
        except ConnectionResetError:
            _LOGGER.debug("Client connection reset")
        except Exception as e:
            _LOGGER.debug("Client error: %s", e)
        finally:
            self.clients.remove(client_writer)
            self._client_write_buffers.pop(id(client_writer), None)
            if not client_writer.is_closing():
                client_writer.close()
                try:
                    await client_writer.wait_closed()
                except Exception:
                    pass

            _LOGGER.debug("Client left, remaining clients: %d", len(self.clients))

            if not self.clients:
                _LOGGER.debug("Last client disconnected, stopping server")
                self.stop()

    async def recv(self):
        """Receive data from target and buffer it for distribution."""
        consecutive_errors = 0

        try:
            _LOGGER.debug("Starting packet reception")
            while (
                not self._stop_requested
                and self.target_reader
                and not self.target_reader.at_eof()
            ):
                try:
                    # Read header from the target server
                    header = await asyncio.wait_for(
                        self.target_reader.read(9), timeout=PACKET_TIMEOUT
                    )

                    if len(header) < 9:
                        if len(header) == 0:
                            consecutive_errors += 1
                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                _LOGGER.warning("Too many empty reads, ending stream")
                                break
                            await asyncio.sleep(0.05)
                            continue
                        consecutive_errors += 1
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            break
                        continue

                    # Reset error counter on successful header read
                    consecutive_errors = 0

                    msgtype = header[0]
                    payload_length = int.from_bytes(header[5:9], byteorder="big")

                    if payload_length <= 0 or payload_length > 65535:
                        continue

                    # Read payload with timeout
                    try:
                        payload_data = await asyncio.wait_for(
                            self._read_payload(payload_length), timeout=PAYLOAD_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        _LOGGER.debug("Timeout reading payload")
                        continue

                    if payload_data is None or len(payload_data) < payload_length:
                        self.health.packet_dropped()
                        continue

                    # Only buffer video packets (msgtype 0x00) starting with 0x47
                    if msgtype != 0x00 or payload_data[0] != 0x47:
                        continue

                    # Add to buffer instead of sending directly
                    await self.packet_buffer.put(payload_data)
                    self.health.packet_received()

                except asyncio.TimeoutError:
                    consecutive_errors += 1
                    _LOGGER.debug(
                        "Timeout waiting for data (%d/%d)",
                        consecutive_errors,
                        MAX_CONSECUTIVE_ERRORS,
                    )
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        break

        except ssl.SSLError as e:
            if e.reason != "APPLICATION_DATA_AFTER_CLOSE_NOTIFY":
                _LOGGER.warning("SSL error: %s", e)
        except Exception as e:
            _LOGGER.warning("Receive error: %s", e)
        finally:
            _LOGGER.debug("Packet reception ended")

    async def _distribute_packets(self):
        """Distribute buffered packets to clients with batching and slow client handling."""
        # Wait for buffer to fill initially for smoother start
        initial_wait = 0
        while (
            not self.packet_buffer.is_ready
            and not self._stop_requested
            and initial_wait < 3.0
        ):
            await asyncio.sleep(0.1)
            initial_wait += 0.1

        _LOGGER.debug(
            "Starting packet distribution (buffer size: %d)", self.packet_buffer.size
        )

        while not self._stop_requested:
            # Wait for data with timeout
            has_data = await self.packet_buffer.wait_for_data(timeout=1.0)

            if not has_data:
                continue

            # Get batch of packets for efficient distribution
            packets = await self.packet_buffer.get_batch(max_count=10)

            if not packets:
                continue

            # Combine packets for batch write
            combined_data = b"".join(packets)

            # Send to all clients
            for writer in list(self.clients):
                if writer.is_closing():
                    continue

                client_id = id(writer)
                pending = self._client_write_buffers.get(client_id, 0)

                # Handle slow clients by dropping packets
                if pending > SLOW_CLIENT_THRESHOLD:
                    self.health.packet_dropped()
                    _LOGGER.debug(
                        "Dropping packets for slow client (pending: %d)", pending
                    )
                    continue

                try:
                    writer.write(combined_data)
                    self._client_write_buffers[client_id] = pending + len(packets)

                    # Drain periodically to prevent buffer buildup
                    if pending > 20:
                        await writer.drain()
                        self._client_write_buffers[client_id] = 0

                except Exception as e:
                    _LOGGER.debug("Error writing to client: %s", e)

            # Small yield to prevent blocking
            await asyncio.sleep(0)

    async def _read_payload(self, length):
        """Read payload with handling for incomplete data."""
        data = bytearray()
        remaining = length

        while remaining > 0:
            chunk = await self.target_reader.read(
                min(remaining, 16384)
            )  # Larger chunks
            if not chunk:
                break
            data.extend(chunk)
            remaining -= len(chunk)

        return bytes(data) if len(data) == length else None

    async def send(self):
        """Send keep-alive and latency-stats messages to the server."""
        latency_stats_packet = bytes(
            [
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
        )
        tick_count = 0
        sequence = 0

        try:
            while (
                not self._stop_requested
                and self.target_writer
                and not self.target_writer.is_closing()
            ):
                # Send full keepalive packet every 10 seconds
                if tick_count % KEEPALIVE_FULL_INTERVAL == 0:
                    sequence += 1
                    sequence_bytes = sequence.to_bytes(4, byteorder="big")
                    keepalive_packet = (
                        bytes([0x0A]) + sequence_bytes + bytes([0x00, 0x00, 0x00, 0x00])
                    )
                    self.target_writer.write(keepalive_packet)

                # Always send latency stats
                self.target_writer.write(latency_stats_packet)
                await self.target_writer.drain()

                tick_count += 1
                await asyncio.sleep(KEEPALIVE_INTERVAL)

        except Exception as e:
            _LOGGER.debug("Keepalive error: %s", e)
        finally:
            if self.target_reader:
                self.target_reader.feed_eof()
            _LOGGER.debug("Keepalive sender ended")

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
                try:
                    response = await asyncio.wait_for(
                        api.request_command_status(
                            self.camera.sync.blink,
                            self.camera.network_id,
                            self.command_id,
                        ),
                        timeout=10.0,
                    )
                except asyncio.TimeoutError:
                    _LOGGER.debug("Command API poll timeout, continuing")
                    await asyncio.sleep(self.polling_interval)
                    continue

                if response.get("status_code", 0) != 908:
                    _LOGGER.warning(
                        "Command API returned unexpected status: %s", response
                    )
                    break

                for commands in response.get("commands", []):
                    if commands.get("id") == self.command_id:
                        state_condition = commands.get("state_condition")
                        if state_condition not in ("new", "running"):
                            _LOGGER.debug(
                                "Command state changed to %s, ending", state_condition
                            )
                            return

                await asyncio.sleep(self.polling_interval)

        except Exception as e:
            _LOGGER.debug("Polling error: %s", e)
        finally:
            _LOGGER.debug("Command polling ended")
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
        self.packet_buffer.clear()

        _LOGGER.debug("Stopping stream, closing connections")

        if self.server and self.server.is_serving():
            self.server.close()

        if self.target_writer and not self.target_writer.is_closing():
            self.target_writer.close()

        for writer in self.clients:
            if not writer.is_closing():
                writer.close()

        _LOGGER.debug("Stream stopped")


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
