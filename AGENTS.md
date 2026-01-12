# AGENTS.md

This file provides guidance for AI coding agents working on this project.

## Project Overview

**blinkliveview** is a command-line tool for streaming live video from Blink cameras. It supports Blink's 2FA authentication and can run as an on-demand stream server for Home Assistant integration.

### Key Features
- Stream live video from Blink cameras to local ffplay
- Server mode for Home Assistant integration (TCP stream + HTTP snapshot endpoint)
- 2FA authentication with credential persistence
- Resilient streaming with auto-retry on connection failures
- Docker support for containerized deployment

## Project Structure

```
blinkliveview/
├── blinkliveview/           # Main package
│   ├── __init__.py
│   ├── cli.py               # CLI entry point, argument parsing, auth, streaming logic
│   └── livestream.py        # Resilient livestream wrapper for handling unstable streams
├── main.py                  # Alternative entry point
├── pyproject.toml           # Project configuration (uv/hatch)
├── Dockerfile               # Container configuration
├── README.md                # User documentation
└── uv.lock                  # Dependency lock file
```

## Technology Stack

- **Python 3.10+** - Minimum required version
- **uv** - Package manager and virtual environment tool
- **blinkpy** - Blink API library for camera access
- **aiohttp** - Async HTTP client/server
- **asyncio** - Core async runtime for stream handling
- **Ruff** - Linter (config in `.ruff_cache/`)

## Key Components

### cli.py (1200+ lines)
The main module containing:
- `get_args()` - CLI argument parsing
- `authenticate()` - Blink authentication with 2FA support
- `stream_camera()` - Local playback via ffplay with retry logic
- `serve_camera()` - On-demand TCP stream server for Home Assistant
- `OnDemandStreamServer` - TCP server that starts streams when clients connect
- `OnDemandLiveStream` - Forwards Blink stream data to connected clients

### livestream.py (~600 lines)
- `PacketBuffer` - Thread-safe buffer for smoothing network jitter
- `ConnectionHealth` - Monitor connection health and detect issues proactively
- `ResilientLiveStream` - Optimized wrapper for handling unstable Blink streams with:
  - Packet buffering to smooth network jitter (50 packet buffer)
  - Automatic reconnection with exponential backoff (up to 5 attempts)
  - Connection health monitoring with proactive reconnection
  - Optimized socket buffer sizes (256KB) for better throughput
  - Batched client writes to reduce I/O overhead
  - Slow client detection and graceful packet dropping
  - Keep-alive and latency stats packets
  - Command API polling with timeout protection

## Development Commands

```bash
# Install dependencies
uv sync

# Run the CLI
uv run blinklive --help
uv run blinklive --list                    # List cameras
uv run blinklive -c "Camera Name"          # Stream to ffplay
uv run blinklive --serve -c "Camera Name"  # Start server mode

# Lint code
uv run ruff check .

# Build Docker image
docker build -t blinkliveview .
```

## Code Style Guidelines

- Use `async/await` for all I/O operations
- Follow existing patterns for error handling with verbose mode support
- Use type hints for function signatures
- Log timestamps with `_get_timestamp()` helper in server mode
- Prefer `asyncio.wait_for()` with timeouts for network operations

## Important Implementation Details

### Blink Stream Protocol
- Streams use `immis://` protocol over TLS
- Authentication uses custom binary header format (see `_get_auth_header()`)
- Video packets have msgtype `0x00` and start with `0x47` (TS sync byte)
- Keep-alive packets must be sent every 10 seconds

### Credential Storage
- Credentials stored in `~/.config/blinkliveview/credentials.json`
- Session tokens auto-refresh every 30 minutes in server mode

### Stream Limitations
- Blink limits live streams to ~5 minutes per session
- Server mode auto-restarts stream when clients remain connected
- Retry logic handles premature stream termination

## Testing Considerations

- No test suite currently exists
- When adding tests, consider mocking the Blink API (`blinkpy`)
- Test both server mode and direct streaming modes
- Handle 2FA flow in test scenarios

## Common Tasks

### Adding a new CLI option
1. Add argument in `get_args()` in `cli.py`
2. Pass to relevant functions (`stream_camera()`, `serve_camera()`, etc.)
3. Update README.md usage section

### Modifying stream handling
1. Changes to packet processing go in `livestream.py` (`recv()` method)
2. Changes to server behavior go in `cli.py` (`OnDemandStreamServer` class)

### Updating dependencies
1. Edit `pyproject.toml`
2. Run `uv sync` to update `uv.lock`

## Stream Optimization Details

The streaming implementation includes several optimizations for stability:

### Packet Buffering
- 50-packet buffer smooths network jitter
- Low watermark (10 packets) before starting distribution
- Prevents stuttering from momentary network hiccups

### Automatic Reconnection
- Exponential backoff: 1s, 2s, 4s, 8s, 16s (max 30s)
- Up to 5 reconnection attempts before giving up
- Health monitoring triggers proactive reconnection

### Slow Client Handling
- Tracks pending writes per client
- Drops packets for clients with >100 pending packets
- Prevents slow clients from blocking stream

### Socket Optimization
- 256KB socket buffers for better throughput
- TCP keepalive enabled on connections
- Larger read chunks (16KB) for efficiency

### Timeout Configuration
- Packet timeout: 15 seconds
- Payload read timeout: 8 seconds
- Max consecutive errors: 15 (increased tolerance)
