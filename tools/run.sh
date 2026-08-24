#!/usr/bin/env bash
# 고정 측정 환경에서 이 저장소의 명령을 실행하는 래퍼.
#
# 쓰는 법 (Windows 셸에서, 저장소 루트에서):
#   MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl bash tools/run.sh python -m pytest -q
#
# 이 저장소의 수치는 전부 이 환경에서 나와야 한다. 다른 python으로 잰 값은
# 논문에 쓸 수 없다 — 이유는 HANDOFF.md의 "고정 측정 환경" 절에 있다.

set -uo pipefail

ENV_ROOT=/opt/conda/envs/e1

# 저장소 위치는 스크립트 자신의 위치에서 구한다. 예전 판은 경로가 하드코딩돼
# 있어서 저장소를 옮기면 조용히 엉뚱한 곳에서 실행됐다.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)

# 고정 환경이 없으면 여기서 멈춘다. 이 검사가 없으면 아래 PATH 대입이 아무
# 효과 없이 지나가고 시스템 python으로 측정이 돌아간다 — 결과는 그럴듯하게
# 나오지만 논문에 쓸 수 없는 값이다. 조용히 틀리느니 시끄럽게 죽는 편이 낫다.
if [ ! -x "$ENV_ROOT/bin/python" ]; then
    echo "고정 환경을 찾을 수 없다: $ENV_ROOT/bin/python" >&2
    echo "WSL2 안에서 실행 중인지, conda env 'e1'이 있는지 확인할 것." >&2
    exit 1
fi

# PATH 대입의 따옴표는 반드시 있어야 한다. Windows PATH가 WSL로 넘어올 때
# "/mnt/c/Program Files/..." 같이 공백이 든 항목이 섞이는데, 따옴표가 없으면
# 단어분할이 일어나 export가 "not a valid identifier"로 실패한다. set -e가
# 없으므로 스크립트는 계속 진행하고 PATH만 원래 값으로 남는다 — 즉 위의
# 검사를 통과하고도 고정 환경이 아닌 python에서 측정이 돌아간다.
export CUDA_HOME="$ENV_ROOT"
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
export CC=gcc-11 CXX=g++-11

# 저장소 모듈을 import할 수 있게 한다. 하위 디렉터리의 스크립트를 직접 돌리면
# sys.path[0]이 그 스크립트의 디렉터리가 되어 bench/ models/ 를 못 찾는다.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
exec "$@"
