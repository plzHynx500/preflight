"""MODULE-01 CanaryEngine 1~2단계 — config만 조회해 랜덤 초기화 모델·더미 입력을 구성한다.

가중치는 다운로드하지 않는다 (docs/architecture.md §5 MODULE-01 참고).

이 모듈의 함수는 전부 **자식 프로세스 안에서만** 호출된다. 부모(CLI)가 torch를
import하는 순간 프로세스 격리가 무력화되므로(docs/contracts/canary-api.md 참고),
torch/transformers import는 반드시 함수 안에서 한다 — 모듈 최상단에 두지 말 것.
"""

from __future__ import annotations

import re

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

# 4bit 변환에서 빼는 레이어. 폴백의 LoRA 부착도 같은 목록을 쓴다 — 두 경로가 "베이스로
# 취급하는 레이어"의 범위를 다르게 잡으면 폴백이 재는 대상이 4bit 경로와 달라진다(#75).
_MODULES_NOT_CONVERTED = ("lm_head",)


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
    동일한 폴백 철학으로 베이스를 평범한 `nn.Linear`로 대체하고 그 사실을
    `quant_backend`로 알린다 — 실패해도 여기서 죽지 않는다. **폴백 모델도 `device`가
    가리키는 곳으로 올린다**(#66) — 안 올리면 입력과 device가 어긋나 폴백이 발동하는
    순간 오히려 RuntimeError로 죽어, 안전장치가 없는 것과 같아진다.

    **폴백도 QLoRA의 *모양*은 유지한다(#75)** — 베이스를 얼리고(`requires_grad_(False)`)
    LoRA 어댑터만 학습 대상으로 붙인다. 양자화(4bit)는 bitsandbytes 없이 못 하지만
    "베이스는 얼고 어댑터만 학습한다"는 부분은 torch만으로 된다. 안 그러면 폴백이 fp32
    **전체 파인튜닝**이 되어 파라미터당 16바이트(가중치 4 + gradient 4 + AdamW 상태 8)가
    들고 — 8B급이면 약 128GB — 위 "QLoRA는 옵션이 아니라 고정 가정"과 정면으로 어긋난다.
    즉 폴백이 발동하는 순간 진단이 재는 대상 자체가 사용자의 계획과 달라진다. 얼리면
    gradient·옵티마이저 상태가 어댑터에만 붙어 파라미터당 약 4바이트(fp32 가중치)로
    내려간다.

    가중치 자체는 여전히 fp32로 통째로 올라가므로(8B 기준 약 30GB) 아주 큰 모델은
    이 경로에서도 못 버틸 수 있다 — "4bit이 안 되는 환경"이라는 신호로는 유효하지만,
    폴백이 만능은 아니라는 뜻이다.

    `(model, config, quant_backend, quant_fallback_reason)`을 돌려준다 — `config`는 `build_dummy_input()`이
    토큰 ID를 만들 때 필요한 `vocab_size`를 담고 있고(docs/contracts/canary-api.md
    참고), `quant_backend`는 W9에서 `run_canary_check()` 반환 스키마에 채워 넣을 값이다.

    4bit 가중치는 uint8로 packed되어 `numel()` 기준 파라미터 수가 실제의 절반으로
    보인다 — 리포트에 파라미터 수를 찍을 일이 있으면 `config`에서 계산해야 한다.
    """
    config = _load_config(model_name)
    model, quant_backend, fallback_reason = _build_qlora_model(config, device)
    return model, config, quant_backend, fallback_reason


class ModelConfigError(RuntimeError):
    """config 조회 실패 중 **원인을 특정할 수 있는** 경우에만 쓴다.

    worker가 이 예외를 traceback으로 포장해 `error_log`에 담고, 리포트가 그 꼬리를
    화면에 보여준다 — 그래서 메시지 자체가 사용자에게 그대로 읽힌다(#62).
    """


def _load_config(model_name: str):
    """`AutoConfig.from_pretrained()` — 실패 원인을 사용자가 읽을 수 있는 말로 바꾼다.

    HF Hub는 **없는 저장소에도 404가 아니라 401을 돌려준다.** transformers가 그
    401을 3단으로 체인해서 올리면 error_log가 57줄이 되고, 화면에는 마지막 줄인
    "If this is a private repository, make sure to pass a token…"만 남는다 —
    모델명을 잘못 친 사용자가 토큰을 찾으러 가는 결과가 된다(#62 실측).

    그래서 여기서 원인을 확정해 짧은 로그로 바꾼다. `from None`으로 체인을 끊는
    것이 핵심이다 — 원본 예외를 `__cause__`로 달아두면 `traceback.format_exc()`가
    3단 체인을 그대로 다시 찍어 고친 의미가 없어진다. 원본 체인은 `--json`에서도
    사라지지만, 버리는 것은 **이미 우리가 원인을 아는** 경우의 401·token 잡음뿐이다
    (#62의 요지가 그 잡음이 사용자를 엉뚱한 곳으로 보낸다는 것이었다).

    **모르는 실패는 건드리지 않고 그대로 올린다.** 오타로 단정하는 것이 이 이슈의
    버그였으므로, 시그니처가 맞는 경우에만 말을 바꾼다.
    """
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(model_name)
    except Exception as exc:  # 아는 원인만 말을 바꾸고, 나머지는 아래에서 그대로 올린다
        message = _config_error_message(model_name, exc)
        if message is None:
            raise
        raise ModelConfigError(message) from None


def _config_error_message(model_name: str, exc: Exception) -> str | None:
    """알려진 config 조회 실패면 한국어 한 줄, 아니면 None.

    huggingface_hub을 import하지 않고 **예외 타입 이름**으로 판별한다 — 부모가 아닌
    자식 프로세스라 import 자체는 가능하지만, hf_hub은 하드 의존성이 아니고 예외
    클래스의 위치가 버전마다 옮겨 다녀서(`huggingface_hub.utils` →
    `huggingface_hub.errors`) 이름 비교가 오히려 안정적이다.
    """
    names = {base.__name__ for base in type(exc).__mro__}
    text = str(exc)

    # 타입만 보면 놓친다 — transformers가 `GatedRepoError`를 잡아 `OSError`로 다시
    # 던지는 경로가 있어(`raise OSError(...) from e`) `__mro__`에 원래 타입이 남지
    # 않는다. 실제로 gated 모델이 "원인 미상 오류(모델명 오타 등)"로 나갔다(#154).
    if "GatedRepoError" in names or "gated repo" in text.lower():
        return (
            f"접근이 제한된 모델: {model_name}"
            " (HF Hub에서 라이선스 동의 또는 접근 권한이 필요하다 — hf auth login)"
        )
    # 설치된 transformers가 모르는 아키텍처. 예외 본문에 조치(`pip install --upgrade
    # transformers`)가 이미 적혀 있는데 화면에는 "오타 등"으로만 나갔다(#154).
    if "does not recognize this architecture" in text:
        model_type = _quoted_model_type(text)
        detail = f"model_type={model_type}" if model_type else "model_type 미상"
        return (
            f"설치된 transformers가 모르는 모델 구조: {model_name} ({detail})"
            " — transformers를 올리거나(pip install --upgrade transformers),"
            " 아주 새 모델이면 아직 지원 릴리스가 없을 수 있다"
        )
    if "RepositoryNotFoundError" in names or "is not a valid model identifier" in text:
        return (
            f"모델을 찾을 수 없음: {model_name}"
            " (HF Hub에 해당 저장소 없음 — 모델명 오타 또는 비공개 저장소)"
        )
    if "HFValidationError" in names:
        return f"모델명 형식이 올바르지 않음: {model_name} (기대 형식: <org>/<name>)"
    if (
        "LocalEntryNotFoundError" in names
        or "OfflineModeIsEnabled" in names
        or "Check your internet connection" in text
    ):
        return (
            f"네트워크에 연결할 수 없고 로컬 캐시에도 없음: {model_name}"
            " (config.json을 받지 못했다 — 연결 확인 후 다시 실행)"
        )
    return None


def _quoted_model_type(text: str) -> str | None:
    """transformers 메시지에서 백틱으로 감싼 model_type을 뽑는다 (#154).

    "has model type `minimax_music3` but Transformers does not recognize ..."
    형태다. 못 찾으면 None — 문구가 바뀌어도 안내 자체는 나가야 한다.
    """
    match = re.search(r"model type `([^`]+)`", text)
    return match.group(1) if match else None


#: `vocab_size`가 최상위에 없을 때 훑어볼 하위 config 이름들 (#154).
#: 멀티모달·통합 config가 텍스트 설정을 아래로 내리면서 이름이 제각각이다.
_TEXT_CONFIG_ATTRS = ("text_config", "llm_config", "language_model", "decoder")


def resolve_vocab_size(config) -> int:
    """더미 토큰 ID를 만들 때 쓸 `vocab_size`를 찾는다 (#154).

    예전에는 `config.vocab_size`를 그냥 읽었다. 요즘 주력 모델들이 텍스트 설정을
    하위 config로 내리면서(`Gemma4UnifiedConfig`·`Qwen3_5Config` 등 통합 config)
    최상위에 `vocab_size`가 없고, 그대로 `AttributeError`로 터져 화면에는
    "원인 미상 오류 (모델명 오타 등)"이 나갔다 — 오타가 아닌데 오타를 찾게 만든다.

    **`get_text_config()`를 먼저 쓴다.** transformers가 "이 config의 텍스트 부분"을
    돌려주려고 둔 공식 API라, 하위 config 이름이 모델마다 달라도 여기서 흡수된다.
    없거나 실패하는 버전을 위해 최상위 → 알려진 하위 이름 순으로 물러선다.

    끝내 못 찾으면 `ModelConfigError`로 **왜 못 하는지**를 말한다. 이 canary는 토큰
    ID를 입력으로 넣는 텍스트 LLM 전제라(architecture.md §6-01), 순수 비전·오디오
    모델은 진단 대상이 아니다.
    """
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            size = getattr(getter(), "vocab_size", None)
        except Exception:  # noqa: BLE001 - 공식 API가 없거나 깨진 버전은 아래로 물러선다
            size = None
        if size:
            return size

    size = getattr(config, "vocab_size", None)
    if size:
        return size

    for attr in _TEXT_CONFIG_ATTRS:
        size = getattr(getattr(config, attr, None), "vocab_size", None)
        if size:
            return size

    raise ModelConfigError(
        f"이 모델의 config에서 vocab_size를 찾지 못했다 ({type(config).__name__})"
        " — 토큰 입력을 받는 텍스트 LLM만 모델 체크를 할 수 있다"
    )


def _build_qlora_model(config, device: str):
    """4bit 양자화 + LoRA 어댑터가 적용된 랜덤 초기화 모델.

    실패하면 **얼린 `nn.Linear` 베이스 + LoRA**로 폴백한다 — 양자화만 빠지고 "베이스는
    얼고 어댑터만 학습한다"는 모양은 그대로다(#75).

    `replace_with_bnb_linear`는 `transformers.integrations.bitsandbytes`의 **비공개
    API**다 — transformers 버전을 올릴 때 이 함수의 존재/시그니처를 확인해야 한다
    (2026-08-06 기준 5.14.1에서는 동작 확인됨).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.integrations.bitsandbytes import replace_with_bnb_linear

        model, quant_backend = _materialize_qlora(
            config, device, torch, AutoModelForCausalLM, replace_with_bnb_linear
        )
        return model, quant_backend, None
    except Exception as error:  # noqa: BLE001 - 실패 종류와 무관하게 nn.Linear 폴백한다
        # **왜 폴백했는지 남긴다**(#147). 예전에는 이 except가 예외를 통째로 삼켜서,
        # 우리 코드의 버그(#134: RoPE inv_freq가 meta로 남아 forward에서 죽던 것)까지
        # 화면에는 "4bit 레이어 구성 실패 → 폴백"으로만 보였다. 읽는 사람은 "이 환경이
        # 4bit을 못 쓰는구나"로 받아들일 뿐, 도구 자신의 결함을 의심할 단서가 없었다.
        fallback_reason = describe_fallback_cause(error)

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_config(config)
        # 베이스를 얼리고 LoRA만 붙여 QLoRA의 모양을 유지한다(#75). 어댑터는 아직
        # CPU에 만들어지고, 바로 아래 .to(device)가 베이스와 함께 옮긴다.
        _freeze_base_and_attach_lora(model)
        # 폴백 모델도 **요청받은 device로 올린다**. 빼먹으면 모델만 CPU에 남아,
        # 같은 device로 만들어진 입력(cuda)과 어긋나 forward에서 RuntimeError로
        # 죽는다 — 폴백은 "bitsandbytes가 없어도 진단은 계속한다"는 안전장치인데
        # 발동하는 순간 도구가 부서졌다(#66). 형제 함수 build_minimal_canary_model도
        # 같은 이유로 무조건 .to(device)를 부른다.
        #
        # torch를 다시 import하지 않고 device 문자열을 그대로 넘긴다 — 이 분기는
        # `import torch` 실패로도 들어올 수 있어서, 여기서 torch를 또 부르면 폴백
        # 자체가 같은 이유로 죽는다. nn.Module.to()는 문자열을 그대로 받는다.
        model = model.to(device)
        return model, QUANT_BACKEND_FALLBACK, fallback_reason


def _freeze_base_and_attach_lora(model) -> bool:
    """폴백 모델의 베이스 `nn.Linear`를 얼리고 LoRA 어댑터만 학습 대상으로 붙인다(#75).

    4bit 경로가 `_materialize_qlora` 끝에서 하는 일과 같고, 대상 타입만
    `Linear4bit` → `nn.Linear`로 넓힌 것이다. 양자화는 bitsandbytes 없이 못 하지만
    "베이스는 얼고 어댑터만 학습한다"는 부분은 torch만으로 되므로, 폴백에서도 실제
    학습이 타는 연산 경로를 그대로 태울 수 있다.

    **`import torch`가 죽어도 여기서 폴백을 깨뜨리지 않는다.** 이 함수를 부르는
    분기는 `import torch` 실패로도 들어올 수 있어서, 실패하면 동결을 되돌리고
    조용히 물러선다 — 안전장치가 안전장치를 부수면 안 된다.

    **어댑터가 하나도 안 붙으면 동결을 되돌린다.** 학습 대상이 0개인 모델을 넘기면
    `worker._execute_canary_cycle`의 `AdamW([])`가
    `ValueError: optimizer got an empty parameter list`로 죽어, 메모리를 아끼려다
    진단 자체를 잃는다.

    실제로 얼렸는지를 bool로 돌려준다.
    """
    try:
        import torch

        for param in model.parameters():
            param.requires_grad_(False)
        attached = _attach_manual_lora(
            model, torch, torch.nn.Linear, skip_names=_MODULES_NOT_CONVERTED
        )
    except Exception:  # noqa: BLE001 - 폴백의 폴백이라 실패해도 진단은 계속한다
        attached = 0

    if attached:
        return True

    for param in model.parameters():
        param.requires_grad_(True)
    return False


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
        model,
        modules_to_not_convert=list(_MODULES_NOT_CONVERTED),
        quantization_config=bnb_config,
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

        # **버퍼도 같은 자리에서 실체화한다.** 파라미터만 채우면 `register_buffer`로
        # 등록된 것들이 meta로 남아, forward에서 그걸 쓰는 순간
        # `NotImplementedError: Cannot copy out of meta tensor`로 죽는다(#134).
        #
        # `with torch.device("meta")`는 그 블록 안에서 만들어지는 **모든** 텐서를
        # meta로 보낸다 — 파라미터와 버퍼를 구분하지 않는다. 그래서 실체화하는 쪽도
        # 둘 다 훑어야 한다.
        #
        # RoPE 계열(Llama·Mistral·Qwen·Gemma)이 위치 주파수를 `inv_freq` 버퍼로 갖고,
        # 그게 정확히 이 구멍에 빠졌다. GPT-2 계열은 위치 정보가 `nn.Embedding`
        # (=파라미터)이라 우연히 멀쩡했다.
        #
        # 값은 0이다. **VRAM은 값이 아니라 모양·자료형으로 정해지므로 측정에 영향이
        # 없고**, 가중치도 이미 랜덤이라 이 canary가 정답을 계산하는 일은 애초에 없다.
        # `inv_freq`가 0이면 회전이 없을 뿐 `cos`/`sin`은 정상이라 forward가 돈다.
        #
        # **미검토 사항(#134)**: 0이 수학적으로 특별한 값이라, 어떤 모델의 버퍼가
        # 나눗셈 분모로 쓰이면 `inf`가 될 수 있다. 전수 확인은 하지 않았다 —
        # `running_var`류는 `eps`가 막아주고 요즘 마스크는 함수로 계산되지만,
        # 구체적 반례가 나오면 값 선택을 재검토한다. 그때 잡아줄 안전망은 값 자체가
        # 아니라 **실체화 후 forward를 실제로 돌리는 테스트**다.
        #
        # `setattr`로 넣으면 `nn.Module.__setattr__`이 이미 버퍼인 이름을 `_buffers`에
        # 다시 넣어줘서 `persistent` 여부가 유지된다(`inv_freq`는 persistent=False다).
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf is None or buf.device.type != "meta":
                continue
            setattr(module, name, torch.zeros(buf.shape, dtype=buf.dtype, device=device))

    # 베이스는 얼린다 — LoRA 어댑터만 gradient·옵티마이저 상태를 갖는다.
    for param in model.parameters():
        param.requires_grad_(False)

    _attach_manual_lora(model, torch, bnb.nn.Linear4bit, dtype=torch.float16, device=device)
    return model, QUANT_BACKEND_4BIT


class _LoraAdapter:
    """`peft` 없이 Linear4bit 하나에 붙는 LoRA 어댑터 — down→up 두 개의 작은 nn.Linear.

    peft가 하드 의존성이 아니라서(사용자 venv를 preflight 설치가 덮어쓰면 안 됨)
    수동으로 구현한다. `up`을 0으로 초기화하는 건 LoRA 표준 관례다 — 학습 시작
    시점에 어댑터가 항등(원본 그대로)이 되게 해서 초기 forward 값이 베이스 모델과
    같게 만든다(여기선 canary라 학습 품질은 상관없지만 관례를 그대로 따른다).
    """

    def __init__(
        self,
        torch_module,
        in_features: int,
        out_features: int,
        rank: int,
        dtype,
        device: str | None = None,
    ):
        self.down = torch_module.nn.Linear(
            in_features, rank, bias=False, dtype=dtype, device=device
        )
        self.up = torch_module.nn.Linear(rank, out_features, bias=False, dtype=dtype, device=device)
        torch_module.nn.init.zeros_(self.up.weight)

    def __call__(self, module, inputs, output):
        return output + self.up(self.down(inputs[0]))


def _attach_manual_lora(
    model, torch, target_cls, dtype=None, device: str | None = None, skip_names=()
) -> int:
    """`target_cls`인 레이어마다 LoRA 어댑터를 forward hook으로 붙이고 그 개수를 돌려준다.

    어댑터를 각 대상 레이어의 서브모듈로 등록해야(`add_module`) `model.parameters()`로
    순회할 때 잡힌다 — worker.py의 optimizer 구성(`[p for p in model.parameters()
    if p.requires_grad]`, 기본 체크와 동일 패턴)이 이 값을 그대로 찾아 쓴다.

    대상 타입을 인자로 받는 이유는 4bit 경로(`Linear4bit`)와 폴백 경로(`nn.Linear`)가
    같은 부착 로직을 쓰기 때문이다(#75). **두 타입을 하나로 합치면 안 된다** —
    `bnb.nn.Linear4bit`은 `nn.Linear`의 서브클래스라, 4bit 경로를 `nn.Linear`로 훑으면
    변환 대상에서 뺀 `lm_head`에까지 어댑터가 붙는다.

    **대상 목록을 먼저 확정한 뒤 부착한다.** `lora_down`/`lora_up`도 `nn.Linear`라,
    순회하면서 붙이면 방금 만든 어댑터에 또 어댑터가 붙는다.

    `dtype=None`이면 각 대상의 `weight.dtype`을 따른다 — 폴백 모델은 fp32라 4bit
    경로처럼 float16으로 고정하면 hook이 받는 입력과 dtype이 어긋난다.
    """
    targets = [
        module
        for name, module in model.named_modules()
        if isinstance(module, target_cls) and name.rsplit(".", 1)[-1] not in skip_names
    ]
    for module in targets:
        adapter = _LoraAdapter(
            torch,
            module.in_features,
            module.out_features,
            MINIMAL_ADAPTER_RANK,
            module.weight.dtype if dtype is None else dtype,
            device,
        )
        module.add_module("lora_down", adapter.down)
        module.add_module("lora_up", adapter.up)
        module.register_forward_hook(adapter)
    return len(targets)


def build_dummy_input(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """`--model` 경로용 더미 입력 — 토큰 ID(정수) 텐서.

    기본 체크(`build_minimal_canary_input`)는 임베딩이 없어 float 텐서를 바로
    쓰지만, 실제 모델은 임베딩 레이어가 있어 정수 토큰 ID가 필요하고 그 범위는
    `vocab_size`가 정한다 — 원래 시그니처(batch_size, seq_len)에는 없던 값이라
    추가했다(docs/contracts/canary-api.md 참고).
    """
    import torch

    return torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.device(device))


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
    fallback_reason = None
    blocks = []
    for _ in range(MINIMAL_NUM_BLOCKS):
        base, backend, reason = _build_base_linear(MINIMAL_HIDDEN_SIZE, dtype, prefer_4bit)
        # 베이스는 얼린다 — QLoRA와 같이 어댑터만 gradient·옵티마이저 상태를 갖는다.
        for param in base.parameters():
            param.requires_grad_(False)
        quant_backend = backend
        fallback_reason = reason
        blocks.append(_CanaryBlock(base, MINIMAL_HIDDEN_SIZE, MINIMAL_ADAPTER_RANK, dtype))

    model = nn.Sequential(*blocks)
    # Linear4bit은 CUDA로 옮기는 이 시점에 실제로 양자화된다.
    model = model.to(torch.device(device))
    return model, quant_backend, fallback_reason


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
    """베이스 레이어 한 장을 `(layer, quant_backend, 폴백 사유)`로 돌려준다.

    `prefer_4bit=False`는 CPU 기준선 재시도처럼 **일부러** 4bit을 안 쓰는 경우라
    사유가 None이다 — 실패한 적이 없으니 설명할 것도 없다(#147).
    """
    from torch import nn

    layer, reason = _try_build_4bit_linear(hidden, dtype) if prefer_4bit else (None, None)
    if layer is not None:
        return layer, QUANT_BACKEND_4BIT, None
    return nn.Linear(hidden, hidden, bias=False, dtype=dtype), QUANT_BACKEND_FALLBACK, reason


def describe_fallback_cause(error: BaseException) -> str:
    """폴백 사유 한 줄 — 예외 **종류와 메시지만** 남긴다 (#147).

    트레이스백은 싣지 않는다. 이 값은 판정이 아니라 "왜 4bit 대신 nn.Linear로
    쟀는가"를 밝히는 한 줄이고, 트레이스백이 필요한 실패는 이미 `error_log`가
    맡고 있다.
    """
    return f"{type(error).__name__}: {error}"


def _try_build_4bit_linear(hidden: int, dtype):
    """bitsandbytes 4bit 레이어를 `(layer, 폴백 사유)`로 돌려준다.

    만들 수 없으면 `(None, "<예외 종류>: <메시지>")`다. bitsandbytes 미설치·구버전·
    빌드 문제 어느 쪽이든 폴백이 정상 동작이므로 여기서 죽이지 않는다. 다만 **왜
    폴백했는지는 남긴다** — 예전에는 예외를 통째로 삼켜서, 우리 코드의 버그까지
    화면에는 그냥 "폴백했다"로만 보였다(#147; #134가 그렇게 가려져 있었다).
    """
    try:
        from bitsandbytes.nn import Linear4bit

        layer = Linear4bit(hidden, hidden, bias=False, compute_dtype=dtype, quant_type="nf4")
        return layer, None
    except Exception as error:  # noqa: BLE001 - 실패 종류와 무관하게 폴백한다
        return None, describe_fallback_cause(error)
