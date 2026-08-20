"""테스트용 VOC 형식 인스턴스 마스크 PNG 작성기.

팔레트를 명시하지 않고 P 모드로 저장하면 Pillow가 실제로 쓰인 색만 남기고
인덱스를 다시 매긴다. 고정 환경(Pillow 12.3.0) 실측:

    저장 {0, 1}      -> 다시 읽으면 {0, 1}       (우연히 살아남는다)
    저장 {0, 1, 2}   -> 다시 읽으면 {0, 1}       인스턴스 2가 사라진다
    저장 {0, 1, 255} -> 다시 읽으면 {0, 1}       void가 사라진다

값이 밝기가 아니라 id인 마스크에서 이것은 조용한 데이터 손상이고, 그 위에
쌓은 테스트는 아무것도 검증하지 않으면서 통과한다. 실제 VOC 파일은 768바이트
팔레트를 갖고 저장되므로 이 함수가 그 구조를 맞춘다.
"""
from pathlib import Path

import numpy as np
from PIL import Image


def write_mask_png(path: Path, mask: np.ndarray) -> Path:
    """인스턴스 마스크를 실제 VOC와 같은 팔레트 PNG로 저장하고 경로를 돌려준다."""
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="P")
    image.putpalette([value // 3 for value in range(768)])
    image.save(path)
    return Path(path)
