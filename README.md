# System One container

A text-only System One API for Cloudflare Clef models. The default model is `Cloudflare/clef-flash`. GitHub Actions builds the container and publishes it to GitHub Container Registry (GHCR).

## Run

The host needs an NVIDIA GPU, a driver compatible with CUDA 12.8, Docker and
NVIDIA Container Toolkit. Replace `<owner>/<repo>` with your repository name.

```sh
docker run -d --name systemone --gpus all --shm-size=2g \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v systemone-model-cache:/data/huggingface \
  ghcr.io/<owner>/<repo>:latest
```

The first run downloads the model. Later runs reuse the mounted cache.
Wait for loading to finish, then check health:

```sh
docker logs -f systemone
# In another terminal:
curl http://127.0.0.1:8000/health
```

The API listens on localhost and has no authentication. For remote access, use
private connectivity or an authenticated proxy.

## Make a request

Send requests to `POST /v1/systemone`. The request's `model` must match the
configured model.

For Pydantic AI, use `SystemOneModel` with model `Cloudflare/clef-flash` and
provider base URL `http://127.0.0.1:8000/v1`.

## Configuration

Set environment variables with Docker's `-e` option, before the image name.
Settings are validated at startup; `/health` reports the active configuration.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SYSTEMONE_BACKEND` | `clef` | Backend; only `clef` is supported |
| `SYSTEMONE_MODEL` | `Cloudflare/clef-flash` | Hugging Face model repository |
| `SYSTEMONE_REVISION` | `fde727a287004204b7518dcc983fe64379776712` | Pinned revision of the default model |
| `SYSTEMONE_MAX_LENGTH` | `4096` | Token limit |
| `HF_HOME` | `/data/huggingface` | Model cache directory |
| `HF_TOKEN` | Unset | Token for restricted repositories |
| `HF_HUB_OFFLINE` | Unset | Set to `1` to use cached files only |

Changing the model requires an explicit revision and a container restart:

```sh
-e SYSTEMONE_MODEL=Cloudflare/clef \
-e SYSTEMONE_REVISION=<reviewed-model-commit>
```

Weights and `joint_schema_model.py` are downloaded at startup. The server executes
that model-supplied helper, so choose a reviewed revision. Compatible Clef models
can share this image if the GPU has enough memory. Other model families need a
new loader and backend validation/dispatch; dependency changes need a new image.

## Build and validation

```sh
docker build -t systemone-server .
```