#!/usr/bin/env bash
# E4의 12 run을 분리된 프로세스로 띄운다. 중단된 캠페인을 이어갈 때도 같은 명령이다.
#
# 쓰는 법 (Windows 셸에서, 저장소 루트에서):
#   MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl bash tools/run_e4.sh
#
# 가드는 넓은 쪽으로 틀린다 — 애매하면 띄우지 않고 거부한다. 잘못 거부하는 것은
# 사람이 보고 넘기면 되지만, 두 번 띄우면 같은 체크포인트를 두 프로세스가 덮어써
# 캠페인을 잃는다.
#
# 이 스크립트가 하는 일은 셋뿐이다 — 이미 돌고 있으면 두 번 띄우지 않고, 호출한
# 셸에서 프로세스를 떼어내고, 로그를 파일로 남긴다. 학습 자체의 재개는
# experiments/e4_ablation.py가 한다(완료된 run은 건너뛰고, 진행 중이던 run은
# 체크포인트에서 이어간다).

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/results/e4"
LOG="$LOG_DIR/run.log"
PID_FILE="$LOG_DIR/run.pid"

# 두 번 띄우면 같은 체크포인트를 두 프로세스가 덮어쓴다. 그쪽이 캠페인을 잃는
# 가장 빠른 길이므로 여기서 막는다.
if pgrep -f 'python.*experiments\.e4_ablation' >/dev/null; then
    echo "이미 돌고 있다:" >&2
    pgrep -af 'experiments.e4_ablation' >&2
    echo "중단하려면 kill 한 뒤 같은 명령으로 다시 띄우면 이어진다." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# setsid로 프로세스 그룹을 떼어낸다. 이걸 안 하면 띄운 셸(또는 그 셸을 띄운
# 도구)이 끝날 때 함께 죽는다. -u는 로그가 실시간으로 쌓이게 한다.
setsid nohup bash "$SCRIPT_DIR/run.sh" python -u -m experiments.e4_ablation \
    >> "$LOG" 2>&1 < /dev/null &

sleep 2
PID=$(pgrep -f 'python.*experiments\.e4_ablation' | head -1)
if [ -z "$PID" ]; then
    echo "띄우지 못했다. $LOG 를 확인할 것." >&2
    exit 1
fi
echo "$PID" > "$PID_FILE"

echo "시작됨 pid=$PID"
echo "로그:    tail -f $LOG"
echo "진행:    cat $REPO_ROOT/results/e4/runs.csv"
echo "중단:    kill $PID   (다시 띄우면 이어진다)"
