# Setup (pyproject + uv)

This project now manages dependencies only in `pyproject.toml`.
`requirements*.txt` is no longer used.

## Local setup

```bash
# Optional: use Python 3.11 (recommended)
uv python install 3.11

# Create/update env from pyproject
uv sync --extra inference

# Optional heavy extras
# uv sync --extra inference --extra p3d
# uv sync --extra inference --extra dev
```

## Docker setup

```bash
docker compose build fastsam3d
docker compose run --rm fastsam3d
uv sync --extra inference
```

## Checkpoints

- SAM3D checkpoints: `checkpoints/hf/*`
- MoGe checkpoint: `checkpoints/moge-vitl/model.pt`

