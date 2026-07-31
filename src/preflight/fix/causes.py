"""원인 시그니처 → 원인 분류. 안정된 파이썬 속성(torch.version.cuda,
bitsandbytes.cextension.lib.compiled_with_cuda, sys.version_info 등)으로 판별하고
nvidia-smi 텍스트 파싱은 쓰지 않는다 (docs/architecture.md §5 MODULE-01 참고).

FIX 문구 자체는 초안 단계다 — docs/architecture.md §6-04 표 참고.
"""

from __future__ import annotations


def classify_cause(check_result: dict) -> str:
    raise NotImplementedError
