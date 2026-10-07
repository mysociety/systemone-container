FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/huggingface

# Same dependency versions as the successful Exoscale A5000 experiment.
RUN uv pip install --no-cache --python /venv/bin/python \
    --torch-backend=cu128 torch==2.11.0 torchvision==0.26.0
COPY requirements.txt /app/requirements.txt
RUN uv pip install --no-cache --python /venv/bin/python -r /app/requirements.txt

WORKDIR /app
COPY serve.py .
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=15m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
