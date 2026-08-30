"""저장소용 그림과 논문용 그림의 차이를 한곳에서 정한다.

크기(figsize)는 모듈마다 패널 배치가 달라 각 모듈이 스스로 정하고, 여기서는
**어떤 style이 있는지**와 dpi만 공유한다. 이름 검증을 공유해야 한 모듈에만
새 style이 생기는 일이 없다.

기본값은 언제나 "repo"다. 기존 호출자(results/*/의 그림을 만드는 코드와 그
테스트)가 모르는 사이에 논문용 크기로 바뀌면 안 된다.
"""
STYLES = ("repo", "paper")

# 논문은 벡터 PDF로 나가지만 e2의 ERF 히트맵은 래스터라 dpi가 그대로 남는다.
_DPI = {"repo": 150, "paper": 300}


def check(style: str) -> str:
    if style not in STYLES:
        raise ValueError(
            f"알 수 없는 style '{style}'. 사용 가능: {', '.join(STYLES)}"
        )
    return style


def dpi(style: str) -> int:
    return _DPI[check(style)]
