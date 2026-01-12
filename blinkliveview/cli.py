#!/usr/bin/env python3
"""Command line interface for Blink Live View."""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from aiohttp import ClientSession, web
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth, BlinkTwoFARequiredError

from blinkliveview.livestream import init_resilient_livestream


# Default config directory
CONFIG_DIR = Path.home() / ".config" / "blinkliveview"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="blinklive",
        description="Stream live video from Blink cameras using ffplay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  blinklive                     # Interactive mode - list cameras and select one
  blinklive --list              # List all available cameras
  blinklive --camera "Front Door"  # Stream from specific camera
  blinklive --logout            # Clear saved credentials
  blinklive --2fa CODE          # Provide 2FA code directly

Server mode (for Home Assistant):
  blinklive --serve -c "Front Door"           # Start server on port 5000
  blinklive --serve -c "Front Door" --port 8554  # Custom port
  blinklive --serve -c "Front Door" --host 192.168.1.100  # Bind to specific IP
        """,
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=str,
        help="Name of the camera to stream (case-insensitive)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available cameras and exit",
    )
    parser.add_argument(
        "--logout",
        action="store_true",
        help="Clear saved credentials and exit",
    )
    parser.add_argument(
        "--2fa",
        "--twofa",
        dest="twofa",
        type=str,
        help="Provide 2FA code directly (skip interactive prompt)",
    )
    parser.add_argument(
        "--username",
        "-u",
        type=str,
        help="Blink account username/email",
    )
    parser.add_argument(
        "--password",
        "-p",
        type=str,
        help="Blink account password",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save credentials to disk",
    )
    parser.add_argument(
        "--serve",
        "-s",
        action="store_true",
        help="Run as a persistent stream server (for Home Assistant integration)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for stream server (default: 5000, use with --serve)",
    )
    parser.add_argument(
        "--snapshot-port",
        type=int,
        default=8080,
        help="Port for HTTP snapshot server (default: 8080, use with --serve)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind stream server (default: 0.0.0.0, use with --serve)",
    )
    parser.add_argument(
        "--ffplay-args",
        type=str,
        default="",
        help="Additional arguments to pass to ffplay",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts if stream fails (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


async def load_credentials() -> dict | None:
    """Load saved credentials from disk."""
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        with open(CREDENTIALS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


async def save_credentials(blink: Blink, save: bool = True):
    """Save credentials to disk."""
    if not save:
        return
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    await blink.save(str(CREDENTIALS_FILE))
    print(f"Credentials saved to {CREDENTIALS_FILE}")


def clear_credentials():
    """Remove saved credentials."""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()
        print(f"Credentials removed from {CREDENTIALS_FILE}")
    else:
        print("No saved credentials found")


async def authenticate(
    session: ClientSession,
    username: str | None = None,
    password: str | None = None,
    twofa_code: str | None = None,
    verbose: bool = False,
) -> Blink:
    """Authenticate with Blink servers, handling 2FA if required."""
    blink = Blink(session=session)

    # Try to load saved credentials first
    saved_creds = await load_credentials()

    if saved_creds:
        if verbose:
            print("Found saved credentials, attempting to authenticate...")
        auth = Auth(saved_creds, no_prompt=True, session=session)
        blink.auth = auth
        try:
            await blink.start()
            if verbose:
                print("Successfully authenticated with saved credentials")
            return blink
        except Exception as e:
            if verbose:
                print(f"Saved credentials failed: {e}")
            print("Saved credentials expired or invalid, re-authenticating...")

    # Get credentials from user if not provided
    if not username:
        username = input("Blink username (email): ").strip()
    if not password:
        import getpass

        password = getpass.getpass("Blink password: ")

    # Create auth with user credentials
    auth = Auth(
        {"username": username, "password": password},
        no_prompt=True,
        session=session,
    )
    blink.auth = auth

    # Attempt to start - this may raise 2FA exception
    try:
        await blink.start()
    except BlinkTwoFARequiredError:
        print("\n2FA verification required!")
        print("A verification PIN has been sent to your Blink account email.")
        print("(Check your inbox and spam folder)")

        if not twofa_code:
            twofa_code = input("Enter 2FA PIN from email: ").strip()

        # Use the Blink.send_2fa_code method which handles the full setup flow
        success = await blink.send_2fa_code(twofa_code)
        if not success:
            raise RuntimeError("2FA verification failed")

        print("2FA verification successful!")

    return blink


async def list_cameras(blink: Blink):
    """List all available cameras."""
    print("\nAvailable cameras:")
    print("-" * 50)

    for idx, (name, camera) in enumerate(blink.cameras.items(), 1):
        sync_module = camera.sync.name if camera.sync else "Unknown"
        camera_type = camera.product_type or "Unknown"
        armed = "Armed" if camera.arm else "Disarmed"
        print(f"  {idx}. {name}")
        print(f"     Type: {camera_type} | Sync: {sync_module} | Status: {armed}")

    print("-" * 50)
    print(f"Total: {len(blink.cameras)} camera(s)")


async def select_camera(blink: Blink) -> str | None:
    """Interactively select a camera."""
    cameras = list(blink.cameras.keys())

    if not cameras:
        print("No cameras found!")
        return None

    if len(cameras) == 1:
        print(f"Auto-selecting only camera: {cameras[0]}")
        return cameras[0]

    print("\nSelect a camera to stream:")
    for idx, name in enumerate(cameras, 1):
        print(f"  {idx}. {name}")

    while True:
        try:
            choice = input("\nEnter camera number (or 'q' to quit): ").strip()
            if choice.lower() == "q":
                return None
            idx = int(choice)
            if 1 <= idx <= len(cameras):
                return cameras[idx - 1]
            print(f"Please enter a number between 1 and {len(cameras)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def find_camera(blink: Blink, camera_name: str) -> str | None:
    """Find camera by name (case-insensitive)."""
    camera_name_lower = camera_name.lower()
    for name in blink.cameras.keys():
        if name.lower() == camera_name_lower:
            return name

    # Try partial match
    matches = [
        name for name in blink.cameras.keys() if camera_name_lower in name.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"Multiple cameras match '{camera_name}': {matches}")
        return None

    return None


async def stream_camera(
    blink: Blink,
    camera_name: str,
    ffplay_args: str,
    verbose: bool = False,
    max_retries: int = 3,
):
    """Stream from a camera using ffplay with auto-retry on failure."""
    camera = blink.cameras.get(camera_name)
    if not camera:
        print(f"Camera '{camera_name}' not found!")
        return

    retries = 0
    while retries <= max_retries:
        if retries > 0:
            print(f"\nRetrying stream (attempt {retries + 1}/{max_retries + 1})...")
            await asyncio.sleep(2)  # Wait before retry

        print(f"\nInitializing livestream for '{camera_name}'...")

        livestream = None
        feed_task = None
        ffplay_proc = None

        try:
            # Initialize the resilient livestream (handles incomplete packets better)
            livestream = await init_resilient_livestream(camera)

            # Start the local server
            await livestream.start(host="127.0.0.1", port=0)
            stream_url = livestream.url

            if verbose:
                print(f"Local stream server started at: {stream_url}")

            print(f"Starting live stream from '{camera_name}'...")
            print("Press Ctrl+C to stop streaming\n")

            # Build ffplay command with robust settings for unstable streams
            ffplay_cmd = [
                "ffplay",
                "-i",
                stream_url,
                "-fflags",
                "+genpts+discardcorrupt",
                "-flags",
                "low_delay",
                "-framedrop",
                "-avioflags",
                "direct",
                "-probesize",
                "32",
                "-analyzeduration",
                "0",
                "-sync",
                "ext",
                "-infbuf",
                "-loglevel",
                "warning",
            ]

            # Add user-specified args (but filter out duplicates)
            if ffplay_args:
                user_args = ffplay_args.split()
                for arg in user_args:
                    if arg not in ffplay_cmd:
                        ffplay_cmd.append(arg)

            if verbose:
                print(f"Running: {' '.join(ffplay_cmd)}")

            # Start the stream feed in background
            feed_task = asyncio.create_task(livestream.feed())

            # Wait a moment for the stream to initialize
            await asyncio.sleep(1.5)

            # Suppress SDL debug messages
            env = os.environ.copy()
            env["SDL_LOG_PRIORITY"] = "critical"

            # Start ffplay as subprocess
            ffplay_proc = subprocess.Popen(
                ffplay_cmd,
                stdout=subprocess.DEVNULL if not verbose else None,
                stderr=subprocess.DEVNULL if not verbose else None,
                env=env,
            )

            # Wait for either ffplay to exit or the feed to end
            stream_duration = 0
            while ffplay_proc.poll() is None and not feed_task.done():
                await asyncio.sleep(0.5)
                stream_duration += 0.5

            # Check if stream ended prematurely (less than 5 seconds)
            ffplay_exit_code = ffplay_proc.poll()
            feed_ended_early = feed_task.done() and stream_duration < 5

            # Cleanup current attempt
            if ffplay_proc.poll() is None:
                ffplay_proc.terminate()
                try:
                    ffplay_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    ffplay_proc.kill()

            livestream.stop()

            if feed_task and not feed_task.done():
                feed_task.cancel()
                try:
                    await feed_task
                except asyncio.CancelledError:
                    pass

            # If stream ran for a reasonable time or user closed it, don't retry
            if stream_duration >= 5 and not feed_ended_early:
                print("\nStream ended.")
                return

            # Stream ended prematurely - retry if we have attempts left
            if retries < max_retries:
                print(f"\nStream ended prematurely after {stream_duration:.1f}s")
                retries += 1
                continue
            else:
                print(f"\nStream failed after {max_retries + 1} attempts.")
                return

        except NotImplementedError as e:
            print(f"Livestream not supported for this camera: {e}")
            return
        except FileNotFoundError:
            print("Error: ffplay not found. Please install ffmpeg/ffplay.")
            print("  macOS: brew install ffmpeg")
            print("  Ubuntu/Debian: sudo apt install ffmpeg")
            print("  Windows: Download from https://ffmpeg.org/download.html")
            return
        except Exception as e:
            if verbose:
                import traceback

                traceback.print_exc()

            # Cleanup on error
            if ffplay_proc and ffplay_proc.poll() is None:
                ffplay_proc.terminate()
            if livestream:
                livestream.stop()
            if feed_task and not feed_task.done():
                feed_task.cancel()
                try:
                    await feed_task
                except asyncio.CancelledError:
                    pass

            if retries < max_retries:
                print(f"\nStream error: {e}")
                retries += 1
                continue
            else:
                print(f"\nStream failed: {e}")
                return


async def serve_camera(
    blink: Blink,
    camera_name: str,
    host: str = "0.0.0.0",
    port: int = 5000,
    snapshot_port: int = 8080,
    verbose: bool = False,
    save_credentials: bool = True,
):
    """Run as an on-demand stream server for Home Assistant integration.

    The stream is only started when a client connects, and stops when
    all clients disconnect. This respects Blink's streaming limits.
    """
    camera = blink.cameras.get(camera_name)
    if not camera:
        print(f"Camera '{camera_name}' not found!")
        return

    print(f"\n{'=' * 60}")
    print(f"Blink Live Stream Server (On-Demand)")
    print(f"{'=' * 60}")
    print(f"Camera: {camera_name}")
    print(f"Stream URL: tcp://{host}:{port}")
    print(f"Snapshot URL: http://{host}:{snapshot_port}/snapshot")
    print(f"")
    print(f"NOTE: Stream starts when a client connects and stops when")
    print(f"      all clients disconnect. This respects Blink's ~5 min")
    print(f"      streaming limit and reduces API usage.")
    print(f"{'=' * 60}")
    print(f"\nHome Assistant configuration:")
    print(f"")
    print(f"  # configuration.yaml")
    print(f"  camera:")
    print(f"    - platform: generic")
    print(f"      name: {camera_name}")
    print(f"      stream_source: tcp://<this-machine-ip>:{port}")
    print(f"      still_image_url: http://<this-machine-ip>:{snapshot_port}/snapshot")
    print(f"")
    print(f"{'=' * 60}")
    print(f"Press Ctrl+C to stop the server")
    print(f"{'=' * 60}\n")

    # Create the on-demand server
    server = OnDemandStreamServer(
        blink, camera, host, port, snapshot_port, verbose, save_credentials
    )

    try:
        await server.run()
    except asyncio.CancelledError:
        print(f"\n[{_get_timestamp()}] Server shutting down...")
    finally:
        await server.stop()


class OnDemandStreamServer:
    """Server that starts Blink stream on-demand when clients connect."""

    # Refresh session every 30 minutes to keep tokens alive
    SESSION_REFRESH_INTERVAL = 30 * 60

    def __init__(
        self,
        blink: Blink,
        camera,
        host: str,
        port: int,
        snapshot_port: int,
        verbose: bool,
        save_credentials: bool = True,
    ):
        self.blink = blink
        self.camera = camera
        self.host = host
        self.port = port
        self.snapshot_port = snapshot_port
        self.verbose = verbose
        self.save_credentials = save_credentials

        self.server = None
        self.http_server = None
        self.http_runner = None
        self.clients = []
        self.livestream = None
        self.feed_task = None
        self._lock = asyncio.Lock()
        self._stopping = False
        self._refresh_task = None

    async def run(self):
        """Start the TCP server and HTTP snapshot server."""
        # Start TCP stream server
        self.server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )

        # Start HTTP snapshot server
        await self._start_http_server()

        print(
            f"[{_get_timestamp()}] Stream server listening on tcp://{self.host}:{self.port}"
        )
        print(
            f"[{_get_timestamp()}] Snapshot server listening on http://{self.host}:{self.snapshot_port}/snapshot"
        )
        print(
            f"[{_get_timestamp()}] Waiting for clients (stream starts on first connection)..."
        )

        # Start background task to keep session alive
        self._refresh_task = asyncio.create_task(self._keep_session_alive())

        try:
            async with self.server:
                await self.server.serve_forever()
        finally:
            if self._refresh_task:
                self._refresh_task.cancel()
                try:
                    await self._refresh_task
                except asyncio.CancelledError:
                    pass
            await self._stop_http_server()

    async def _start_http_server(self):
        """Start the HTTP server for snapshot endpoint."""
        app = web.Application()
        app.router.add_get("/snapshot", self._handle_snapshot)
        app.router.add_get("/", self._handle_health)

        self.http_runner = web.AppRunner(app)
        await self.http_runner.setup()
        site = web.TCPSite(self.http_runner, self.host, self.snapshot_port)
        await site.start()

    async def _stop_http_server(self):
        """Stop the HTTP server."""
        if self.http_runner:
            await self.http_runner.cleanup()
            self.http_runner = None

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        return web.json_response(
            {
                "status": "ok",
                "camera": self.camera.name,
                "stream_clients": len(self.clients),
            }
        )

    async def _handle_snapshot(self, request: web.Request) -> web.Response:
        """Handle snapshot requests - returns JPEG image from camera thumbnail."""
        try:
            # First try to get a fresh thumbnail
            await self.camera.snap_picture()
            await asyncio.sleep(1)  # Give Blink time to generate thumbnail
            await self.blink.refresh()

            # Get the cached image
            image_data = self.camera.image_from_cache
            if image_data:
                return web.Response(
                    body=image_data,
                    content_type="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
                )

            # Fallback: try to get thumbnail directly
            if hasattr(self.camera, "thumbnail") and self.camera.thumbnail:
                session = self.blink.auth.session
                async with session.get(self.camera.thumbnail) as resp:
                    if resp.status == 200:
                        image_data = await resp.read()
                        return web.Response(
                            body=image_data,
                            content_type="image/jpeg",
                            headers={
                                "Cache-Control": "no-cache, no-store, must-revalidate"
                            },
                        )

            return web.Response(status=404, text="No snapshot available")

        except Exception as e:
            if self.verbose:
                print(f"[{_get_timestamp()}] Snapshot error: {e}")
            return web.Response(status=500, text=f"Error getting snapshot: {e}")

    async def _keep_session_alive(self):
        """Periodically refresh the Blink session to prevent token expiration."""
        while not self._stopping:
            try:
                await asyncio.sleep(self.SESSION_REFRESH_INTERVAL)

                if self._stopping:
                    break

                # Only refresh if no active stream (stream refresh happens in _start_stream)
                async with self._lock:
                    if not self.clients:
                        print(f"[{_get_timestamp()}] Refreshing Blink session...")
                        try:
                            await self.blink.refresh()
                            # Save updated credentials
                            if self.save_credentials:
                                await self.blink.save(str(CREDENTIALS_FILE))
                            print(
                                f"[{_get_timestamp()}] Session refreshed successfully"
                            )
                        except Exception as e:
                            print(f"[{_get_timestamp()}] Session refresh failed: {e}")
                            if self.verbose:
                                import traceback

                                traceback.print_exc()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[{_get_timestamp()}] Error in session refresh loop: {e}")

    async def stop(self):
        """Stop the server and cleanup."""
        self._stopping = True

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        await self._stop_stream()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        """Handle a new client connection."""
        client_addr = writer.get_extra_info("peername")
        print(f"[{_get_timestamp()}] Client connected: {client_addr}")

        async with self._lock:
            self.clients.append(writer)

            # Start stream if this is the first client
            if len(self.clients) == 1:
                await self._start_stream()

        try:
            # Keep connection alive and forward stream data
            while not self._stopping and not writer.is_closing():
                # Check if stream is still running
                if self.feed_task and self.feed_task.done():
                    # Stream ended, try to restart it
                    async with self._lock:
                        if self.clients and not self._stopping:
                            print(f"[{_get_timestamp()}] Stream ended, restarting...")
                            await self._stop_stream()
                            await asyncio.sleep(1)
                            await self._start_stream()

                await asyncio.sleep(0.5)

        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            if self.verbose:
                print(f"[{_get_timestamp()}] Client error: {e}")
        finally:
            print(f"[{_get_timestamp()}] Client disconnected: {client_addr}")

            async with self._lock:
                if writer in self.clients:
                    self.clients.remove(writer)

                if not writer.is_closing():
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except Exception:
                        pass

                # Stop stream if no more clients
                if not self.clients:
                    print(
                        f"[{_get_timestamp()}] No clients connected, stopping stream..."
                    )
                    await self._stop_stream()
                    print(f"[{_get_timestamp()}] Waiting for clients...")

    async def _start_stream(self):
        """Start the Blink livestream."""
        if self.livestream or self.feed_task:
            return

        try:
            print(f"[{_get_timestamp()}] Starting Blink livestream...")

            # Refresh blink session
            await self.blink.refresh()

            # Initialize livestream - using our custom on-demand version
            self.livestream = await self._init_on_demand_livestream()

            # Start the feed
            self.feed_task = asyncio.create_task(self.livestream.feed())

            print(f"[{_get_timestamp()}] Livestream active")

        except Exception as e:
            print(f"[{_get_timestamp()}] Failed to start stream: {e}")
            if self.verbose:
                import traceback

                traceback.print_exc()
            await self._stop_stream()

    async def _stop_stream(self):
        """Stop the Blink livestream."""
        if self.feed_task:
            self.feed_task.cancel()
            try:
                await self.feed_task
            except asyncio.CancelledError:
                pass
            self.feed_task = None

        if self.livestream:
            self.livestream.stop()
            self.livestream = None

    async def _init_on_demand_livestream(self):
        """Initialize an on-demand livestream that forwards to connected clients."""
        from blinkpy import api

        response = await api.request_camera_liveview(
            self.camera.sync.blink,
            self.camera.sync.network_id,
            self.camera.camera_id,
            camera_type=self.camera.camera_type,
        )

        if self.verbose:
            print(f"[{_get_timestamp()}] Liveview API response: {response}")

        if not response:
            raise RuntimeError("Empty response from Blink liveview API")

        if "server" not in response:
            # Check for error message in response
            if "message" in response:
                raise RuntimeError(f"Blink API error: {response['message']}")
            raise RuntimeError(f"Unexpected API response (no 'server' key): {response}")

        if not response["server"].startswith("immis://"):
            raise NotImplementedError(f"Unsupported protocol: {response['server']}")

        return OnDemandLiveStream(self, response)


class OnDemandLiveStream:
    """Livestream that forwards data to the OnDemandStreamServer's clients.

    Optimized for stream stability with:
    - Packet buffering to smooth network jitter
    - Batched client writes for efficiency
    - Slow client detection and graceful degradation
    - Increased error tolerance
    - Connection health monitoring
    """

    # Configuration constants
    BUFFER_SIZE = 50  # Packets to buffer for jitter smoothing
    BUFFER_LOW_WATERMARK = 10  # Start sending when buffer reaches this
    MAX_CONSECUTIVE_ERRORS = 15  # More tolerance before giving up
    PACKET_TIMEOUT = 15.0  # Timeout for receiving packets
    PAYLOAD_TIMEOUT = 8.0  # Timeout for reading payload
    SLOW_CLIENT_THRESHOLD = 100  # Drop packets if client buffer exceeds this
    SOCKET_BUFFER_SIZE = 262144  # 256KB socket buffer

    def __init__(self, server: OnDemandStreamServer, response):
        import collections
        import urllib.parse

        self.server = server
        self.camera = server.camera
        self.command_id = response["command_id"]
        self.polling_interval = response["polling_interval"]
        self.target = urllib.parse.urlparse(response["server"])
        self.target_reader = None
        self.target_writer = None
        self._stop_requested = False

        # Packet buffer for jitter smoothing
        self._packet_buffer = collections.deque(maxlen=self.BUFFER_SIZE)
        self._buffer_lock = asyncio.Lock()
        self._buffer_event = asyncio.Event()

        # Client write tracking for slow client handling
        self._client_pending_writes: dict = {}

        # Statistics
        self._packets_received = 0
        self._packets_forwarded = 0
        self._packets_dropped = 0
        self._last_packet_time = 0.0

    def _get_auth_header(self):
        """Get authentication header for Blink stream."""
        import urllib.parse

        auth_header = bytearray()
        serial_max_length = 16
        token_field_max_length = 64
        conn_id_max_length = 16

        magic_number = [0x00, 0x00, 0x00, 0x28]
        auth_header.extend(magic_number)

        # Serial field
        serial = self.camera.serial
        field_bytes = serial.encode("utf-8")[:serial_max_length].ljust(
            serial_max_length, b"\x00"
        )
        auth_header.extend(len(field_bytes).to_bytes(4, byteorder="big"))
        auth_header.extend(field_bytes)

        # Client ID
        client_id = urllib.parse.parse_qs(self.target.query).get("client_id", [0])[0]
        auth_header.extend(int(client_id).to_bytes(4, byteorder="big"))

        # Static field
        auth_header.extend([0x01, 0x08])

        # Token field (null)
        auth_header.extend(token_field_max_length.to_bytes(4, byteorder="big"))
        auth_header.extend([0x00] * token_field_max_length)

        # Connection ID
        conn_id = self.target.path.split("/")[-1].split("__")[0]
        field_bytes = conn_id.encode("utf-8")[:conn_id_max_length].ljust(
            conn_id_max_length, b"\x00"
        )
        auth_header.extend(len(field_bytes).to_bytes(4, byteorder="big"))
        auth_header.extend(field_bytes)

        # Trailer
        auth_header.extend([0x00, 0x00, 0x00, 0x01])

        return auth_header

    async def feed(self):
        """Connect to Blink and stream data to all connected clients."""
        import socket
        import ssl
        import time

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        print(
            f"[{_get_timestamp()}] Connecting to Blink server: {self.target.hostname}:{self.target.port}"
        )

        try:
            self.target_reader, self.target_writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.target.hostname, self.target.port, ssl=ssl_context
                ),
                timeout=10.0,
            )

            # Optimize socket settings
            sock = self.target_writer.get_extra_info("socket")
            if sock:
                try:
                    sock.setsockopt(
                        socket.SOL_SOCKET, socket.SO_RCVBUF, self.SOCKET_BUFFER_SIZE
                    )
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                except Exception:
                    pass

        except Exception as e:
            print(f"[{_get_timestamp()}] Failed to connect to Blink server: {e}")
            raise

        print(f"[{_get_timestamp()}] Connected to Blink server, sending auth...")

        # Send auth header
        auth_header = self._get_auth_header()
        self.target_writer.write(auth_header)
        await self.target_writer.drain()

        print(f"[{_get_timestamp()}] Auth sent, starting to receive stream data...")

        # Initialize timing
        self._last_packet_time = time.monotonic()

        try:
            await asyncio.gather(
                self._recv(),
                self._distribute_packets(),
                self._send_keepalive(),
                self._poll(),
                self._health_monitor(),
            )
        except Exception as e:
            if self.server.verbose:
                print(f"[{_get_timestamp()}] Stream error: {e}")
        finally:
            self.stop()

    async def _health_monitor(self):
        """Monitor connection health and trigger reconnection if stale."""
        import time

        while not self._stop_requested:
            await asyncio.sleep(5.0)

            if self._stop_requested:
                break

            # Check for stale connection
            seconds_since_packet = time.monotonic() - self._last_packet_time
            if seconds_since_packet > self.PACKET_TIMEOUT:
                print(
                    f"[{_get_timestamp()}] No packets for {seconds_since_packet:.1f}s, "
                    f"connection stale - stopping stream for restart"
                )
                # Stop the stream to trigger server restart
                self.stop()
                break

            # Log periodic stats
            if self._packets_forwarded > 0 and self._packets_forwarded % 1000 == 0:
                drop_rate = (
                    self._packets_dropped
                    / (self._packets_received + self._packets_dropped)
                    * 100
                    if (self._packets_received + self._packets_dropped) > 0
                    else 0
                )
                print(
                    f"[{_get_timestamp()}] Stats: {self._packets_forwarded} forwarded, "
                    f"{self._packets_dropped} dropped ({drop_rate:.1f}%), "
                    f"buffer: {len(self._packet_buffer)}"
                )

    async def _recv(self):
        """Receive data from Blink and buffer for distribution."""
        import time

        consecutive_errors = 0

        while not self._stop_requested and not self.target_reader.at_eof():
            try:
                # Read header
                header = await asyncio.wait_for(
                    self.target_reader.read(9), timeout=self.PACKET_TIMEOUT
                )

                if len(header) < 9:
                    consecutive_errors += 1
                    if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                        print(f"[{_get_timestamp()}] Too many header errors, stopping")
                        break
                    if len(header) == 0:
                        await asyncio.sleep(0.05)
                    continue

                consecutive_errors = 0
                self._packets_received += 1
                self._last_packet_time = time.monotonic()

                msgtype = header[0]
                payload_length = int.from_bytes(header[5:9], byteorder="big")

                if payload_length <= 0 or payload_length > 65535:
                    continue

                # Read payload
                payload = await self._read_exact(payload_length)
                if payload is None:
                    self._packets_dropped += 1
                    continue

                # Only buffer video packets (msgtype 0x00) starting with 0x47
                if msgtype != 0x00 or payload[0] != 0x47:
                    continue

                # Add to buffer
                async with self._buffer_lock:
                    self._packet_buffer.append(payload)
                    self._buffer_event.set()

            except asyncio.TimeoutError:
                consecutive_errors += 1
                if self.server.verbose:
                    print(
                        f"[{_get_timestamp()}] Timeout waiting for data "
                        f"({consecutive_errors}/{self.MAX_CONSECUTIVE_ERRORS})"
                    )
                if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    break
            except Exception as e:
                print(f"[{_get_timestamp()}] Receive error: {e}")
                break

        print(
            f"[{_get_timestamp()}] Stream ended. Received {self._packets_received} packets, "
            f"forwarded {self._packets_forwarded}, dropped {self._packets_dropped}"
        )

    async def _distribute_packets(self):
        """Distribute buffered packets to clients with batching."""
        # Wait for initial buffer fill for smoother start
        initial_wait = 0
        while (
            len(self._packet_buffer) < self.BUFFER_LOW_WATERMARK
            and not self._stop_requested
            and initial_wait < 3.0
        ):
            await asyncio.sleep(0.1)
            initial_wait += 0.1

        if self.server.verbose:
            print(
                f"[{_get_timestamp()}] Starting distribution (buffer: {len(self._packet_buffer)})"
            )

        first_packet_logged = False

        while not self._stop_requested:
            # Wait for data
            try:
                await asyncio.wait_for(self._buffer_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Get batch of packets
            async with self._buffer_lock:
                packets = []
                for _ in range(min(10, len(self._packet_buffer))):
                    if self._packet_buffer:
                        packets.append(self._packet_buffer.popleft())
                if not self._packet_buffer:
                    self._buffer_event.clear()

            if not packets:
                continue

            # Combine packets for batch write
            combined_data = b"".join(packets)
            clients_count = len(self.server.clients)

            # Forward to all clients
            for writer in list(self.server.clients):
                if writer.is_closing():
                    continue

                client_id = id(writer)
                pending = self._client_pending_writes.get(client_id, 0)

                # Handle slow clients
                if pending > self.SLOW_CLIENT_THRESHOLD:
                    self._packets_dropped += len(packets)
                    if self.server.verbose:
                        print(
                            f"[{_get_timestamp()}] Dropping packets for slow client "
                            f"(pending: {pending})"
                        )
                    continue

                try:
                    writer.write(combined_data)
                    self._client_pending_writes[client_id] = pending + len(packets)

                    # Drain periodically
                    if pending > 20:
                        await writer.drain()
                        self._client_pending_writes[client_id] = 0

                except Exception as e:
                    if self.server.verbose:
                        print(f"[{_get_timestamp()}] Error writing to client: {e}")

            self._packets_forwarded += len(packets)

            if not first_packet_logged:
                print(
                    f"[{_get_timestamp()}] First video packet forwarded to "
                    f"{clients_count} client(s)"
                )
                first_packet_logged = True
            elif self._packets_forwarded % 500 == 0:
                print(
                    f"[{_get_timestamp()}] Forwarded {self._packets_forwarded} packets "
                    f"to {clients_count} client(s)"
                )

            await asyncio.sleep(0)

    async def _read_exact(self, length):
        """Read exact number of bytes with larger chunks."""
        data = bytearray()
        remaining = length

        while remaining > 0:
            try:
                chunk = await asyncio.wait_for(
                    self.target_reader.read(min(remaining, 16384)),
                    timeout=self.PAYLOAD_TIMEOUT,
                )
                if not chunk:
                    return None
                data.extend(chunk)
                remaining -= len(chunk)
            except asyncio.TimeoutError:
                return None

        return bytes(data)

    async def _send_keepalive(self):
        """Send keepalive packets to Blink."""
        latency_stats = bytes(
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

        sequence = 0
        every10s = 0

        while (
            not self._stop_requested
            and self.target_writer
            and not self.target_writer.is_closing()
        ):
            try:
                if every10s % 10 == 0:
                    sequence += 1
                    keepalive = (
                        bytes([0x0A])
                        + sequence.to_bytes(4, byteorder="big")
                        + bytes([0x00, 0x00, 0x00, 0x00])
                    )
                    self.target_writer.write(keepalive)

                self.target_writer.write(latency_stats)
                await self.target_writer.drain()

                every10s += 1
                await asyncio.sleep(1)
            except Exception:
                break

    async def _poll(self):
        """Poll Blink command API with timeout protection."""
        from blinkpy import api

        while not self._stop_requested:
            try:
                response = await asyncio.wait_for(
                    api.request_command_status(
                        self.camera.sync.blink, self.camera.network_id, self.command_id
                    ),
                    timeout=10.0,
                )

                if response.get("status_code", 0) != 908:
                    break

                for cmd in response.get("commands", []):
                    if cmd.get("id") == self.command_id:
                        if cmd.get("state_condition") not in ("new", "running"):
                            return

                await asyncio.sleep(self.polling_interval)
            except asyncio.TimeoutError:
                if self.server.verbose:
                    print(f"[{_get_timestamp()}] Command API poll timeout")
                await asyncio.sleep(self.polling_interval)
            except Exception:
                break

        # Notify command done
        try:
            from blinkpy import api

            await api.request_command_done(
                self.camera.sync.blink, self.camera.network_id, self.command_id
            )
        except Exception:
            pass

    def stop(self):
        """Stop the stream and cleanup."""
        self._stop_requested = True
        self._packet_buffer.clear()
        self._buffer_event.set()  # Unblock distribution task

        if self.target_writer and not self.target_writer.is_closing():
            self.target_writer.close()


def _get_timestamp():
    """Get current timestamp for logging."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def async_main():
    """Async main function."""
    args = get_args()

    # Handle logout
    if args.logout:
        clear_credentials()
        return 0

    # Check for ffplay only if not in serve mode
    if not args.serve:
        try:
            subprocess.run(
                ["ffplay", "-version"],
                capture_output=True,
                check=True,
            )
        except FileNotFoundError:
            print("Error: ffplay not found. Please install ffmpeg/ffplay first.")
            print("  macOS: brew install ffmpeg")
            print("  Ubuntu/Debian: sudo apt install ffmpeg")
            print("  Windows: Download from https://ffmpeg.org/download.html")
            return 1

    session = ClientSession()

    try:
        # Authenticate
        print("Connecting to Blink...")
        blink = await authenticate(
            session,
            username=args.username,
            password=args.password,
            twofa_code=args.twofa,
            verbose=args.verbose,
        )

        # Refresh to get latest camera info
        await blink.refresh()

        # Save credentials if requested
        if not args.no_save:
            await save_credentials(blink)

        # Handle list mode
        if args.list:
            await list_cameras(blink)
            return 0

        # Find or select camera
        camera_name = None
        if args.camera:
            camera_name = find_camera(blink, args.camera)
            if not camera_name:
                print(f"Camera '{args.camera}' not found.")
                await list_cameras(blink)
                return 1
        else:
            await list_cameras(blink)
            camera_name = await select_camera(blink)
            if not camera_name:
                return 0

        # Run in server mode or stream mode
        if args.serve:
            await serve_camera(
                blink,
                camera_name,
                args.host,
                args.port,
                args.snapshot_port,
                args.verbose,
                save_credentials=not args.no_save,
            )
        else:
            await stream_camera(
                blink, camera_name, args.ffplay_args, args.verbose, args.retries
            )

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
    finally:
        await session.close()


def main():
    """Main entry point."""
    try:
        exit_code = asyncio.run(async_main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
