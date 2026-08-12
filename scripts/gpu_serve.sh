#!/usr/bin/env bash
# GPU 컨테이너에서 llama-server 를 띄운다.
#
# 이 스크립트는 GPU가 할당된 컨테이너(gpu1) 안에서 실행한다.
# gpu1 이 cont1 의 네트워크 네임스페이스를 공유하므로(--network container:cont1),
# 여기서 127.0.0.1:$PORT 로 뜬 서버를 cont1 쪽 Claude 세션이 그대로 호출할 수 있다.
#
# 사용:
#   docker exec -d gpu1 bash /workspace/scripts/gpu_serve.sh <gguf경로> [포트] [llama-server경로]
#
# 예:
#   docker exec -d gpu1 bash /workspace/scripts/gpu_serve.sh /workspace/models/Qwen3-8B-Q4_K_M.gguf 8080
set -euo pipefail

# docker exec -d 로 띄우면 표준출력이 버려지므로, 준비 과정도 파일에 남긴다.
mkdir -p /workspace/logs
exec > >(tee -a /workspace/logs/gpu_serve.log) 2>&1
echo "===== $(date '+%F %T') gpu_serve 시작 ====="

GGUF="${1:?사용법: gpu_serve.sh <gguf> [port] [llama-server]}"
PORT="${2:-8080}"

# 이미 떠 있으면 아무것도 하지 않는다.
# 덕분에 이 스크립트를 몇 번을 다시 실행해도 안전하다(포트 충돌·중복 기동 없음).
if curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then
  echo "이미 서버가 떠 있다 → http://127.0.0.1:$PORT (아무것도 하지 않음)"
  exit 0
fi

# llama-server 후보: ① 원 환경 빌드(CUDA) ② 우리가 받은 CPU 빌드 ③ 새로 빌드한 것
# 3번째 인자가 숫자면 컨텍스트 크기로 받는다(경로와 헷갈리기 쉬워서).
LS="${3:-}"
if [[ "$LS" =~ ^[0-9]+$ ]]; then
  CTX="$LS"
  LS=""
fi

# 실행 가능한지(--version 이 실제로 도는지)까지 확인한다.
# 공유 마운트에 있는 CPU 빌드는 다른 컨테이너에서 libgomp 부재로 못 뜨는 일이 있다.
if [ -z "$LS" ]; then
  for cand in \
    /home/dev/llama.cpp/build/bin/llama-server \
    /workspace/tools/llama-cuda/llama-server \
    /workspace/tools/llama.cpp/build/bin/llama-server \
    /workspace/tools/llama-b10360/llama-server
  do
    if [ -x "$cand" ] && "$cand" --version >/dev/null 2>&1; then
      LS="$cand"; break
    fi
    [ -x "$cand" ] && echo "건너뜀(실행 불가): $cand"
  done
fi
[ -n "$LS" ] || { echo "실행 가능한 llama-server 를 찾지 못했다"; exit 1; }

# GPU 빌드인지 확인 — CPU 빌드로 조용히 돌아가면 몇 시간을 낭비하게 된다.
# 주의: llama-server --version 은 백엔드를 출력하지 않는다. 링크된 라이브러리와
# 바이너리 옆 ggml 백엔드 .so 로 판별해야 한다.
BINDIR="$(dirname "$LS")"
if ls "$BINDIR"/libggml-cuda.so* >/dev/null 2>&1 \
   || ldd "$LS" 2>/dev/null | grep -qiE 'libcud(art|a)|libcublas'; then
  echo "CUDA 빌드 확인: $LS"
elif [ -n "${ALLOW_CPU:-}" ]; then
  echo "CPU 빌드로 강행: $LS"
else
  echo "경고: CUDA 빌드로 보이지 않는다 ($LS)"
  echo "      CPU 로 강행하려면 ALLOW_CPU=1 을 붙여 다시 실행하라"
  exit 2
fi

mkdir -p /workspace/logs
LOG="/workspace/logs/server_${PORT}_$(basename "$GGUF" .gguf).log"
# 컨테이너마다 uid 가 달라 공유 마운트의 기존 로그 파일에 못 쓰는 일이 있다.
# 쓸 수 없으면 조용히 죽지 말고 쓸 수 있는 경로로 물러난다.
if ! : > "$LOG" 2>/dev/null; then
  echo "로그 경로에 쓸 수 없어 대체 경로 사용: $LOG"
  LOG="/tmp/server_${PORT}.log"
  : > "$LOG" || { echo "대체 로그도 쓸 수 없다"; exit 1; }
fi

echo "서버: $LS"
echo "모델: $GGUF"
echo "로그: $LOG"

# -c 32768 / -np 4 = 슬롯당 8192 토큰.
# 재서술 파일럿은 인간 원문 전체가 프롬프트에 들어가므로 기존 러너의 -c 16384(슬롯당 4096)로는
# 빠듯하다(원 러너 주석: Qwen3 토크나이저가 한국어를 잘게 쪼개 HTTP 500 유발).
# 컨텍스트는 4번째 인자 또는 CTX 환경변수로 조정 (두 모델을 동시에 띄울 때 줄인다)
CTX="${4:-${CTX:-32768}}"
echo "컨텍스트: $CTX (슬롯 4개 → 슬롯당 $((CTX/4)))"
"$LS" -m "$GGUF" -ngl 99 -c "$CTX" -np 4 \
      --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
SRV=$!
echo "pid=$SRV, 헬스체크 대기..."

for _ in $(seq 1 240); do
  if curl -s -m 3 "http://127.0.0.1:$PORT/health" | grep -q ok; then
    echo "준비 완료 → http://127.0.0.1:$PORT"
    exit 0
  fi
  sleep 5
done

echo "20분 내 준비되지 않음. 로그 확인: $LOG"
exit 1
