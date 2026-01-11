FROM python:3.12-alpine

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY blinkliveview/ blinkliveview/

# Install dependencies
RUN uv sync --no-dev

# Create directory for credentials
RUN mkdir -p /root/.config/blinkliveview

# Default ports
EXPOSE 5000
EXPOSE 8080

# Environment variables for configuration
ENV BLINK_CAMERA=""
ENV BLINK_PORT="5000"
ENV BLINK_SNAPSHOT_PORT="8080"
ENV BLINK_HOST="0.0.0.0"

ENTRYPOINT ["uv", "run", "blinklive", "--serve"]
CMD []
