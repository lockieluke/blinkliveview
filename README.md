# blinkliveview

Command line tool for streaming live video from Blink cameras. Supports Blink's 2FA authentication and can run as an on-demand stream server for Home Assistant integration.

## Requirements

- Python 3.10+
- ffplay (part of ffmpeg) - only required for local playback, not for server mode

## Installation

```bash
# Clone the repository
git clone https://github.com/lockieluke/blinkliveview.git
cd blinkliveview

# Install with uv
uv sync
```

## Usage

### First Run / Authentication

On first run, you'll be prompted for your Blink credentials:

```bash
uv run blinklive
```

A 2FA PIN will be sent to your Blink account email. Enter it when prompted. Credentials are saved to `~/.config/blinkliveview/credentials.json` for future use.

### List Cameras

```bash
uv run blinklive --list
```

### Stream to Local Player

Stream directly to ffplay:

```bash
uv run blinklive -c "Front Door"
```

### Server Mode (for Home Assistant)

Run as an on-demand stream server. The stream only starts when a client connects and stops when all clients disconnect, respecting Blink's ~5 minute streaming limit.

```bash
uv run blinklive --serve -c "Front Door" --port 5000 --snapshot-port 8080
```

This starts two servers:
- **Stream server** (TCP): `tcp://0.0.0.0:5000` - MPEG-TS live video stream
- **Snapshot server** (HTTP): `http://0.0.0.0:8080/snapshot` - JPEG still image

Options:
- `--port PORT` - Port for stream server (default: 5000)
- `--snapshot-port PORT` - Port for HTTP snapshot server (default: 8080)
- `--host HOST` - Host to bind to (default: 0.0.0.0)

### All Options

```
usage: blinklive [-h] [--camera CAMERA] [--list] [--logout] [--2fa TWOFA]
                 [--username USERNAME] [--password PASSWORD] [--no-save]
                 [--serve] [--port PORT] [--snapshot-port PORT] [--host HOST]
                 [--ffplay-args FFPLAY_ARGS] [--retries RETRIES] [--verbose]

Options:
  -h, --help            show this help message and exit
  --camera, -c CAMERA   Name of the camera to stream (case-insensitive)
  --list, -l            List all available cameras and exit
  --logout              Clear saved credentials and exit
  --2fa, --twofa TWOFA  Provide 2FA code directly (skip interactive prompt)
  --username, -u        Blink account username/email
  --password, -p        Blink account password
  --no-save             Don't save credentials to disk
  --serve, -s           Run as a persistent stream server
  --port PORT           Port for stream server (default: 5000)
  --snapshot-port PORT  Port for HTTP snapshot server (default: 8080)
  --host HOST           Host to bind stream server (default: 0.0.0.0)
  --ffplay-args         Additional arguments to pass to ffplay
  --retries RETRIES     Number of retry attempts if stream fails (default: 3)
  --verbose, -v         Enable verbose output
```

## Docker

### Build

```bash
docker build -t blinkliveview .
```

### Initial Setup (Get Credentials)

Run interactively first to authenticate and save credentials:

```bash
docker run -it --rm \
  -v blinklive-config:/root/.config/blinkliveview \
  blinkliveview --list
```

Enter your username, password, and 2FA code when prompted. Credentials will be saved to the Docker volume.

### Run Server

```bash
docker run -d \
  --name blinklive \
  --restart unless-stopped \
  -p 5000:5000 \
  -p 8080:8080 \
  -v blinklive-config:/root/.config/blinkliveview \
  blinkliveview -c "Front Door" --port 5000 --snapshot-port 8080
```

### Docker Compose

```yaml
version: '3.8'
services:
  blinklive:
    build: .
    container_name: blinklive
    restart: unless-stopped
    ports:
      - "5000:5000"
      - "8080:8080"
    volumes:
      - blinklive-config:/root/.config/blinkliveview
    command: ["-c", "Front Door", "--port", "5000", "--snapshot-port", "8080"]

volumes:
  blinklive-config:
```

First run with `docker compose run --rm blinklive --list` to authenticate, then start with `docker compose up -d`.

### Multiple Cameras

Run multiple containers on different ports:

```yaml
version: '3.8'
services:
  blink-front:
    build: .
    restart: unless-stopped
    ports:
      - "5001:5000"
      - "8081:8080"
    volumes:
      - blinklive-config:/root/.config/blinkliveview
    command: ["-c", "Front Door", "--port", "5000", "--snapshot-port", "8080"]

  blink-back:
    build: .
    restart: unless-stopped
    ports:
      - "5002:5000"
      - "8082:8080"
    volumes:
      - blinklive-config:/root/.config/blinkliveview
    command: ["-c", "Back Yard", "--port", "5000", "--snapshot-port", "8080"]

volumes:
  blinklive-config:
```

## Home Assistant Integration

Add to your `configuration.yaml`:

```yaml
camera:
  - platform: generic
    name: Blink Front Door
    stream_source: tcp://192.168.1.100:5000
    still_image_url: http://192.168.1.100:8080/snapshot
```

Replace `192.168.1.100` with the IP of the machine running blinkliveview.

For Docker on the same host as Home Assistant:

```yaml
camera:
  - platform: generic
    name: Blink Front Door
    stream_source: tcp://host.docker.internal:5000
    still_image_url: http://host.docker.internal:8080/snapshot
```

### Snapshot Endpoint

The HTTP snapshot endpoint (`/snapshot`) provides a JPEG still image from the camera. It:
- Requests a fresh thumbnail from Blink
- Returns the cached image with `no-cache` headers
- Also provides a health check at `/` returning JSON status

## Notes

- Blink limits live streams to approximately 5 minutes per session
- The on-demand server automatically restarts the stream if it ends while clients are connected
- Session tokens are refreshed every 30 minutes to prevent expiration
- Credentials are stored in `~/.config/blinkliveview/credentials.json`

## License

MIT
