"""MODULE-01 CanaryEngine 1~2단계 — config만 조회해 랜덤 초기화 모델·더미 입력을 구성한다.

가중치는 다운로드하지 않는다 (docs/architecture.md §5 MODULE-01 참고).

이 모듈의 함수는 전부 **자식 프로세스 안에서만** 호출된다. 부모(CLI)가 torch를
import하는 순간 프로세스 격리가 무력화되므로(docs/contracts/canary-api.md 참고),
torch/transformers import는 반드시 함수 안에서 한다 — 모듈 최상단에 두지 말 것.
"""

from __future__ import annotations

# 기본 체크용 최소 대표 구조의 크기 (docs/architecture.md §3 "기본 체크").
# 특정 모델을 재현하는 것이 목적이 아니라 GPU/드라이버/CUDA 체인이 물리적으로
# 살아있는지만 보므로, 임베딩·attention 없이 4bit 레이어와 어댑터만 둔다.
#
# hidden 크기는 CPU 대비 배수의 신뢰도가 결정한다 — 너무 작으면 GPU·CPU 양쪽 다
# 커널 실행 오버헤드에 묻혀 배수가 흐려진다(docs/architecture.md §6-01). RTX 4070 Ti
# 실측에서 1024는 배수 1.8배로 WARN 임계값(2배)을 밑돌아 정상 환경에 오탐을 냈고,
# 4096은 14배로 충분한 여유가 나왔다. 8B급 모델의 실제 hidden도 4096이다.
MINIMAL_HIDDEN_SIZE = 4096
MINIMAL_NUM_BLOCKS = 2
MINIMAL_ADAPTER_RANK = 16

QUANT_BACKEND_4BIT = "bnb-4bit"
QUANT_BACKEND_FALLBACK = "nn-linear-fallback"


def build_dummy_model(model_name: str):
    """`--model` 경로용 랜덤 초기화 모델을 구성한다.

    가중치는 다운로드하지 않는다 — `AutoConfig.from_pretrained()`로 구조(config.json,
    수 KB)만 조회하고 `from_config()`로 랜덤 초기화한다(docs/architecture.md §5
    MODULE-01). QLoRA(4bit) 래핑 등 학습 설정 세부 옵션은 MVP에서 고정 플래그가
    없어 다루지 않는다(docs/architecture.md §3 "학습 설정 세부 옵션").

    `(model, config)`를 함께 돌려준다 — `build_dummy_input()`이 토큰 ID를 만들려면
    `config.vocab_size`가 필요하기 때문이다(docs/contracts/canary-api.md 참고, 상영님
    지적 사항).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)
    return model, config


def build_dummy_input(batch_size: int, seq_len: int, vocab_size: int):
    """`--model` 경로용 더미 입력 — 토큰 ID(정수) 텐서.

    기본 체크(`build_minimal_canary_input`)는 임베딩이 없어 float 텐서를 바로
    쓰지만, 실제 모델은 임베딩 레이어가 있어 정수 토큰 ID가 필요하고 그 범위는
    `vocab_size`가 정한다 — 원래 시그니처(batch_size, seq_len)에는 없던 값이라
    추가했다(docs/contracts/canary-api.md 참고).
    """
    import torch

    return torch.randint(0, vocab_size, (batch_size, seq_len))


def build_minimal_canary_model(device: str, dtype, prefer_4bit: bool = True):
    """기본 체크용 최소 대표 구조를 만들어 `(model, quant_backend)`로 돌려준다.

    QLoRA와 같은 모양이다 — 4bit로 얼린 베이스 레이어 + 학습되는 작은 어댑터.
    실제 학습이 타는 연산 경로를 그대로 타야 진단이 의미를 갖기 때문이다.
    bitsandbytes가 없거나 4bit 레이어 구성이 실패하면 평범한 `nn.Linear`로
    대체하고 그 사실을 `quant_backend`로 알린다 (docs/architecture.md §6-01).
    """
    import torch
    from torch import nn

    class _CanaryBlock(nn.Module):
        def __init__(self, base: nn.Module, hidden: int, rank: int, block_dtype) -> None:
            super().__init__()
            self.base = base
            self.adapter_down = nn.Linear(hidden, rank, bias=False, dtype=block_dtype)
            self.adapter_up = nn.Linear(rank, hidden, bias=False, dtype=block_dtype)

        def forward(self, x):
            return self.base(x) + self.adapter_up(self.adapter_down(x))

    quant_backend = QUANT_BACKEND_FALLBACK
    blocks = []
    for _ in range(MINIMAL_NUM_BLOCKS):
        base, backend = _build_base_linear(MINIMAL_HIDDEN_SIZE, dtype, prefer_4bit)
        # 베이스는 얼린다 — QLoRA와 같이 어댑터만 gradient·옵티마이저 상태를 갖는다.
        for param in base.parameters():
            param.requires_grad_(False)
        quant_backend = backend
        blocks.append(_CanaryBlock(base, MINIMAL_HIDDEN_SIZE, MINIMAL_ADAPTER_RANK, dtype))

    model = nn.Sequential(*blocks)
    # Linear4bit은 CUDA로 옮기는 이 시점에 실제로 양자화된다.
    model = model.to(torch.device(device))
    return model, quant_backend


def build_minimal_canary_input(batch_size: int, seq_len: int, device: str, dtype):
    """기본 체크용 더미 입력.

    `--model` 경로의 입력은 토큰 ID(정수)라 config의 `vocab_size`가 필요하지만,
    기본 체크 모델에는 임베딩이 없어 임베딩 이후 단계의 float 텐서를 바로 넣는다.
    """
    import torch

    return torch.randn(
        batch_size, seq_len, MINIMAL_HIDDEN_SIZE, device=torch.device(device), dtype=dtype
    )


def _build_base_linear(hidden: int, dtype, prefer_4bit: bool):
    """베이스 레이어 한 장을 만들어 `(layer, quant_backend)`로 돌려준다."""
    from torch import nn

    layer = _try_build_4bit_linear(hidden, dtype) if prefer_4bit else None
    if layer is not None:
        return layer, QUANT_BACKEND_4BIT
    return nn.Linear(hidden, hidden, bias=False, dtype=dtype), QUANT_BACKEND_FALLBACK


def _try_build_4bit_linear(hidden: int, dtype):
    """bitsandbytes 4bit 레이어. 만들 수 없으면 None을 돌려준다.

    bitsandbytes 미설치·구버전·빌드 문제 어느 쪽이든 폴백이 정상 동작이므로 여기서
    죽이지 않는다. 폴백 사실은 호출 측이 `quant_backend`로 위에 알린다.
    """
    try:
        from bitsandbytes.nn import Linear4bit

        return Linear4bit(hidden, hidden, bias=False, compute_dtype=dtype, quant_type="nf4")
    except Exception:  # noqa: BLE001 - 실패 종류와 무관하게 폴백한다
        return None
