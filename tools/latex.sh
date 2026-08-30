#!/usr/bin/env bash
# 논문을 빌드한다. tools/run.sh와 같은 철학이다 - 툴체인이 없으면 죽는다.
#
# 쓰는 법 (Windows 셸에서, 저장소 루트에서):
#   MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl bash tools/latex.sh
#
# 조용히 다른 LaTeX으로 돌아가면 폰트·패키지 판본이 달라져 재현되지 않는다.
# tools/run.sh가 측정에 대해 하는 일을 이 스크립트가 조판에 대해 한다.
#
# tectonic을 쓰는 이유: 단일 바이너리이고 필요한 패키지를 빌드 시점에 받아온다.
# TeX Live 전체(수 GB)를 설치하지 않아도 되고 sudo가 필요 없다. 첫 빌드는
# 패키지를 받느라 느리고 네트워크가 필요하다.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
cd "$REPO_ROOT/paper"

TECTONIC=""
for candidate in "$HOME/bin/tectonic" "$(command -v tectonic 2>/dev/null || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then TECTONIC="$candidate"; break; fi
done

if [ -z "$TECTONIC" ]; then
    echo "tectonic이 없다. 설치 절차는 docs/superpowers/plans/2026-08-30-paper-v2.md Task 1을 볼 것." >&2
    exit 1
fi

exec "$TECTONIC" -X compile main.tex --keep-logs "$@"
