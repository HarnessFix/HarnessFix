FROM python:3.11-slim-bookworm

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir appworld==0.1.3.post1
RUN python3 -m appworld.cli install

RUN mkdir -p /workspace/appworld_root/data \
    && mkdir -p /workspace/appworld_root/experiments \
    && mkdir -p /workspace/output

ENV APPWORLD_ROOT=/workspace/appworld_root
ENV PYTHONWARNINGS=ignore
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["/bin/bash"]
