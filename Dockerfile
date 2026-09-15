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

# 실행 환경 예시로 고정한 PyTorch cu121 라인 (원실험 전체 환경의 lockfile은 아님)
RUN uv venv /opt/venv \
 && VIRTUAL_ENV=/opt/venv uv pip install \
      torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 \
 && chmod -R 777 /opt/venv

COPY requirements.txt /tmp/project-requirements.txt
RUN VIRTUAL_ENV=/opt/venv uv pip install \
      -r /tmp/project-requirements.txt torch==2.5.1

# 코드는 실행 시 /workspace에 마운트한다. 말뭉치·체크포인트·llama-server는 별도 준비.
WORKDIR /workspace
CMD ["bash"]
