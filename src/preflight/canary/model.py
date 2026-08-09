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


def build_dummy_model(model_name: str, device: str = "cuda"):
    """`--model` 경로용 랜덤 초기화 모델을 구성한다.

    가중치는 다운로드하지 않는다 — `AutoConfig.from_pretrained()`로 구조(config.json,
    수 KB)만 조회한다(docs/architecture.md §5 MODULE-01).

    **QLoRA(4bit 양자화 + LoRA 어댑터)를 항상 적용한다** — 옵션이 아니라
    architecture.md §3 "학습 설정 세부 옵션"의 고정 가정이다. 이게 없으면 8B급
    모델 fp32 풀파인튜닝 기준 128GB(가중치 32GB+gradient 32GB+AdamW 상태 64GB)가
    필요해 12GB급 GPU에서는 사실상 항상 OOM이 나서 VRAM 실측 자체가 불가능해진다.

    **v1(폐기): `AutoModelForCausalLM.from_config(config, quantization_config=...)`는
    동작하지 않는다.** 4bit 양자화는 "가중치 파일을 읽으면서" 일어나는 기능이라
    (`from_pretrained`가 디스크에서 한 층씩 읽으며 즉시 4bit으로 압축) 파일을 안 읽는
    `from_config`에는 `quantization_config` 인자 자체가 없다 — 넘기면
    `TypeError: ...__init__() got an unexpected keyword argument 'quantization_config'`.
    게다가 `peft`는 하드 의존성이 아니라 없을 수도 있다. 두 문제 다 이전 구현의
    `except Exception`에 조용히 삼켜져서, `quant_backend`가 사용자 환경과 무관하게
    **항상** `"nn-linear-fallback"`으로 떨어지는 채로 머지 직전까지 안 걸리고 있었다
    (2026-08-06, 상영님이 실측(torch 2.11.0+cu128·transformers 5.14.1·bitsandbytes
    0.50.0)으로 발견 — RTX 4070 Ti에서 재현 스크립트로 직접 검증됨).

    **v2(현재): `meta` 디바이스 위에서 골격만 만들고, 레이어 단위로 랜덤 값을
    채워가며 4bit으로 실체화한다.** 요지는 "파일에서 읽어 채운다"는 `from_pretrained`
    전제를 벗어나, 체크포인트 없이도 레이어 하나 분량(최대 수백 MB)만 RAM에 잠깐
    올렸다가 GPU 4bit로 바로 압축하는 것 — 전체 모델이 fp16으로 통째로 RAM에 존재하는
    순간이 아예 없다. `peft`도 하드 의존성으로 못 쓰므로 LoRA 어댑터는 수동으로
    Linear4bit마다 forward hook으로 붙인다. 상영님이 RTX 4070 Ti·8B급 모델 기준
    RAM 피크 1.75GB · VRAM 최고점 5.77GB로 실측 검증한 방식이다 — 상세 수치와
    구현 스케치는 PR #12 리뷰 코멘트 참고.

    4bit 구성이 실패하면(bitsandbytes 미설치·구버전 등) `build_minimal_canary_model`과
    동일한 폴백 철학으로 평범한 fp32 전체 모델로 대체하고 그 사실을 `quant_backend`로
    알린다 — 실패해도 여기서 죽지 않는다. 단 폴백 시 RAM에 fp32 전체 모델이 통째로
    올라가므로(8B 기준 약 30GB), 실제로 이 경로를 타는 대형 모델은 canary 도구 자체가
    먼저 죽을 수 있다 — 이는 "4bit이 안 되는 환경"이라는 신호로서는 유효하지만,
    쾌적하게 죽지는 않는다는 뜻이다(별도 개선 여지, 지금 범위 밖).

    `(model, config, quant_backend)`를 돌려준다 — `config`는 `build_dummy_input()`이
    토큰 ID를 만들 때 필요한 `vocab_size`를 담고 있고(docs/contracts/canary-api.md
    참고), `quant_backend`는 W9에서 `run_canary_check()` 반환 스키마에 채워 넣을 값이다.

    4bit 가중치는 uint8로 packed되어 `numel()` 기준 파라미터 수가 실제의 절반으로
    보인다 — 리포트에 파라미터 수를 찍을 일이 있으면 `config`에서 계산해야 한다.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    model, quant_backend = _build_qlora_model(config, device)
    return model, config, quant_backend


def _build_qlora_model(config, device: str):
    """4bit 양자화 + LoRA 어댑터가 적용된 랜덤 초기화 모델. 실패하면 fp32로 폴백.

    `replace_with_bnb_linear`는 `transformers.integrations.bitsandbytes`의 **비공개
    API**다 — transformers 버전을 올릴 때 이 함수의 존재/시그니처를 확인해야 한다
    (2026-08-06 기준 5.14.1에서는 동작 확인됨).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.integrations.bitsandbytes import replace_with_bnb_linear

        return _materialize_qlora(
            config, device, torch, AutoModelForCausalLM, replace_with_bnb_linear
        )
    except Exception:  # noqa: BLE001 - 실패 종류와 무관하게 fp32 폴백한다
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_config(config)
        return model, QUANT_BACKEND_FALLBACK


def _materialize_qlora(config, device, torch, AutoModelForCausalLM, replace_with_bnb_linear):
    """meta 골격 → nn.Linear를 Linear4bit로 치환 → 레이어 단위 랜덤 값 실체화 → 수동 LoRA.

    이 순서(치환을 실체화보다 먼저)가 핵심이다 — 아직 meta인 상태에서 구조만
    Linear4bit로 바꿔두면, 그다음 실체화 루프가 "이 파라미터가 4bit로 압축될
    자리인지"를 `isinstance` 하나로 판별할 수 있다. 실체화를 먼저 하면 평범한
    fp16 nn.Linear로 전체 모델이 RAM에 존재하는 순간이 생겨버린다.
    """
    from transformers import BitsAndBytesConfig

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = replace_with_bnb_linear(
        model, modules_to_not_convert=["lm_head"], quantization_config=bnb_config
    )

    import bitsandbytes as bnb

    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if param.device.type != "meta":
                continue
            if isinstance(module, bnb.nn.Linear4bit) and name == "weight":
                # 한 층 분량(최대 수백 MB)만 잠깐 RAM에 만들었다가 4bit으로 압축해 GPU로.
                plain = torch.randn(param.shape, dtype=torch.float16)
                setattr(
                    module,
                    name,
                    bnb.nn.Params4bit(plain, requires_grad=False, quant_type="nf4").to(device),
                )
                del plain
            else:
                setattr(
                    module,
                    name,
                    torch.nn.Parameter(
                        torch.randn(param.shape, dtype=torch.float16, device=device)
                    ),
                )

    # 베이스는 얼린다 — LoRA 어댑터만 gradient·옵티마이저 상태를 갖는다.
    for param in model.parameters():
        param.requires_grad_(False)

    _attach_manual_lora(model, bnb, torch, device)
    return model, QUANT_BACKEND_4BIT


class _LoraAdapter:
    """`peft` 없이 Linear4bit 하나에 붙는 LoRA 어댑터 — down→up 두 개의 작은 nn.Linear.

    peft가 하드 의존성이 아니라서(사용자 venv를 preflight 설치가 덮어쓰면 안 됨)
    수동으로 구현한다. `up`을 0으로 초기화하는 건 LoRA 표준 관례다 — 학습 시작
    시점에 어댑터가 항등(원본 그대로)이 되게 해서 초기 forward 값이 베이스 모델과
    같게 만든다(여기선 canary라 학습 품질은 상관없지만 관례를 그대로 따른다).
    """

    def __init__(
        self, torch_module, in_features: int, out_features: int, rank: int, dtype, device: str
    ):
        self.down = torch_module.nn.Linear(
            in_features, rank, bias=False, dtype=dtype, device=device
        )
        self.up = torch_module.nn.Linear(rank, out_features, bias=False, dtype=dtype, device=device)
        torch_module.nn.init.zeros_(self.up.weight)

    def __call__(self, module, inputs, output):
        return output + self.up(self.down(inputs[0]))


def _attach_manual_lora(model, bnb, torch, device: str) -> None:
    """모든 Linear4bit에 LoRA 어댑터를 forward hook으로 붙인다.

    어댑터를 각 Linear4bit의 서브모듈로 등록해야(`add_module`) `model.parameters()`로
    순회할 때 잡힌다 — worker.py의 optimizer 구성(`[p for p in model.parameters()
    if p.requires_grad]`, 기본 체크와 동일 패턴)이 이 값을 그대로 찾아 쓴다.
    """
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            adapter = _LoraAdapter(
                torch,
                module.in_features,
                module.out_features,
                MINIMAL_ADAPTER_RANK,
                torch.float16,
                device,
            )
            module.add_module("lora_down", adapter.down)
            module.add_module("lora_up", adapter.up)
            module.register_forward_hook(adapter)


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
