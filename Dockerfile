# Use a Python image with uv pre-installed
#FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Add uv depedency
#COPY --from=ghcr.io/astral-sh/uv:0.6.0 /uv /uvx /bin/

# Or use a pre-installed version
 FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Enable bytecode compilation
#ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project into `/app`
WORKDIR /app

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Run the FastAPI application by default
# Uses `fastapi dev` to enable hot-reloading when the `watch` sync occurs
# Uses `--host 0.0.0.0` to allow access from outside the container
CMD ["uv", "run", "streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

