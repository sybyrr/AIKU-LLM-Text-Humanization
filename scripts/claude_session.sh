#!/usr/bin/env bash
# Claude Code 세션을 tmux 안에서 띄우거나, 이미 있으면 거기에 다시 붙는다.
#
# 목적: SSH 가 끊겨도 대화가 죽지 않게 한다.
#   docker exec 로 직접 claude 를 띄우면 SSH 연결이 끊길 때 프로세스가 같이 죽는다.
#   tmux 서버는 컨테이너 안에서 계속 살아 있으므로, 다시 붙기만 하면 이어진다.
#
# 사용 (호스트에서):
#   docker exec -it cont1 bash /workspace/scripts/claude_session.sh
#
# tmux 안에서:
#   빠져나오기(세션은 계속 살아 있음)  Ctrl+B 누른 뒤 D
#   스크롤                              Ctrl+B 누른 뒤 [ , 끝내려면 q
set -euo pipefail

# 한글이 __ 로 깨지지 않게 한다.
# 이 컨테이너는 LANG 이 비어 있어 로케일이 POSIX 다. tmux 는 LC_ALL/LC_CTYPE/LANG 에
# "UTF-8" 이 없으면 클라이언트를 비-UTF-8 모드로 잡고, 표현 못 하는 글자를 셀마다 `_`
# 로 바꿔 버린다(한글은 2칸 폭 → 글자당 `__`). 입력도 같이 깨진다.
# 아래 두 줄이 본 수정이고, tmux -u 는 혹시 로케일이 또 비었을 때를 위한 이중 안전장치다.
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

SESSION="${TMUX_SESSION:-claude}"
WORKDIR="/workspace"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "기존 세션에 다시 붙습니다: $SESSION"
  exec tmux -u attach -t "$SESSION"
fi

echo "새 세션을 만듭니다: $SESSION"
# 셸을 먼저 띄운다 — claude 가 종료돼도 tmux 세션은 남아서 재사용할 수 있다
tmux -u new-session -d -s "$SESSION" -c "$WORKDIR"
# 새로 만드는 창들도 UTF-8 로케일을 물려받게 한다
tmux setenv -g LANG "$LANG"
tmux setenv -g LC_ALL "$LC_ALL"
# 마우스 휠 스크롤 (없으면 Ctrl+B [ 로만 스크롤 가능해 불편하다)
tmux set -g mouse on
tmux set -g history-limit 50000
# 가장 최근 대화를 이어서 시작 (다른 대화를 고르려면 tmux 안에서 claude --resume)
tmux send-keys -t "$SESSION" "claude --continue" C-m
exec tmux -u attach -t "$SESSION"
