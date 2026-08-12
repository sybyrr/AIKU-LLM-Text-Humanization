FROM nvidia/cuda:12.6.3-base-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git ripgrep less jq \
    python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/dev && chmod 777 /home/dev
ENV HOME=/home/dev
ENV PATH="/home/dev/.local/bin:/opt/venv/bin:${PATH}"

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# torch (Pascal sm_60 커널 포함된 cu121 라인)
RUN uv venv /opt/venv \
 && VIRTUAL_ENV=/opt/venv uv pip install \
      torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 \
 && chmod -R 777 /opt/venv

WORKDIR /workspace
CMD ["bash"]