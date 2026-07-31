"""GPU 유무에 따른 조건부 테스트를 위한 공용 픽스처."""

from __future__ import annotations

import pytest


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU가 필요한 테스트")
