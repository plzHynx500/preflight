# ADR-0009: GeForce RTX 50 시리즈(Blackwell)는 드라이버 매핑과 무관하게 cu128 이상을 강제한다

**상태**: Accepted (2026-08-23)
**관련 Issue**: #102 (#84 실환경 재현 검증 중 발견), 관련 ADR: [ADR-0007](0007-driver-version-based-torch-cuda-wheel-selection.md)

## 맥락

ADR-0007(#82)이 `torch_cpu_only_build`의 `fix_command`를 `env.gpu_driver_version`의 major 브랜치 번호로 cu124/cu126/cu130 중 하나를 고르도록 바꿨다. 그런데 이 매핑은 "드라이버가 최신일수록 더 최신 torch를 받게 한다"는 **최신성** 기준으로 설계된 것이지, "이 GPU 아키텍처에 맞는 커널이 그 휠에 있는가"라는 **호환성** 기준이 아니다.

#102가 실측(Windows + RTX 5070 Laptop 8GB, driver 595.79)한 것처럼, RTX 50 시리즈(Blackwell, compute capability sm_120)에서 cu124 휠로 재설치해도 `RuntimeError: CUDA error: no kernel image is available for execution on the device`가 재현된다. 웹 조사(2026-08)로 확인한 사실:

- Blackwell(sm_120) 커널은 PyTorch 2.7.0이 cu128 휠에 처음 넣었다. 그 이전 cu124·cu126 빌드는 sm_120이 세상에 나오기 전에 컴파일된 것이라 드라이버를 아무리 올려도 커널 자체가 없다.
- 이 실패는 드라이버 호환성 문제가 아니라 **휠에 아예 없는 커널을 요청하는** 문제라, ADR-0007의 "재확인이 필요한 시점" 조건(태그 동결·새 CUDA GA)과는 다른 축의 결함이다.

ADR-0007의 드라이버 major 매핑(`_TORCH_CUDA_TAG_BY_MIN_DRIVER_MAJOR`)은 다음과 같다:

```
driver major >= 580  →  cu130
driver major >= 560  →  cu126
그 외                →  cu124 (기본값)
```

RTX 50 시리즈는 드라이버 major 560~579대(cu126 구간)에서도 정상적으로 구동된다 — 즉 Blackwell GPU를 쓰면서 이 구간의 드라이버를 쓰는 사용자는 ADR-0007 이후에도 `no kernel image is available` 오류가 그대로 재현된다. #102 실측 환경(major 595)은 우연히 cu130 구간이라 이미 해결된 것처럼 보였을 뿐이다.

`query_gpu_state()`(`src/preflight/gpu.py`)는 이미 GPU 이름(`name`)을 조회하고 있었지만 `env`에는 실려 있지 않았다(`gpu_free_mb`·`gpu_total_mb`·`gpu_driver_version`만 병합).

## 결정

**`env.gpu_name`을 추가하고, 이름이 GeForce RTX 50 시리즈로 판별되면 드라이버 기반 매핑 결과가 `cu124`·`cu126`일 때 `cu128`로 끌어올린다.** `cu130`(드라이버 major≥580)은 이미 cu128보다 신규 CUDA라 sm_120을 포함하므로 그대로 둔다.

```
driver 기반 태그가 cu130               →  그대로 cu130
driver 기반 태그가 cu124 또는 cu126
    AND GPU 이름이 "RTX 50xx" 패턴     →  cu128로 override
그 외                                  →  driver 기반 태그 그대로
```

판별 정규식은 `RTX 50\d{2}`(대소문자 무시) — GeForce RTX 5050/5060/5070/5080/5090과 그 Ti/Super/Laptop 변형만 잡는다. 구현: `src/preflight/fix/executor.py`의 `_is_blackwell_geforce()` · `_torch_cuda_tag_for_env()`.

**`cu128`을 ADR-0007의 일반 드라이버 매핑 표에 넣지 않고 여기서만 강제하는 이유**: ADR-0007은 cu128이 torch 2.11.0에 동결돼 있어 "최신성" 기준으로는 cu126·cu130보다 손해라고 판단해 후보에서 뺐다. 그 판단 자체는 "GPU가 어떤 아키텍처든 최신 torch를 받는 게 낫다"는 일반 원칙에서는 여전히 맞다. 그러나 Blackwell GPU에게는 "최신 torch를 받되 애초에 그 카드에서 동작하지 않는 것"보다 "약간 오래된 torch라도 실제로 동작하는 것"이 명백히 우선한다 — 정확성(동작 여부)이 최신성(버전 번호)을 이긴다. 이 우선순위 역전은 Blackwell처럼 "커널 자체가 없는" 경우에만 성립하므로 일반 매핑 표를 바꾸지 않고 예외로 처리한다.

**GeForce RTX 50xx만 잡고 데이터센터/프로 Blackwell(B100·B200·GB200·RTX PRO 6000 Blackwell 등)은 범위 밖으로 둔 이유**: #102는 RTX 50 시리즈 소비자 GPU만 실측·보고했다. 데이터센터 카드는 이름 규칙과 실제 배포 환경(주로 Linux, 별도 드라이버 브랜치)이 달라 검증 없이 정규식을 넓히면 오탐 위험이 있다 — 필요해지면 실측 후 별도 Issue로 확장한다.

## 대안

| | 왜 안 택했나 |
|---|---|
| **cu124 기본값을 전체적으로 cu128 이상으로 올린다(이슈 제안 1)** | Blackwell이 아닌 구형 GPU·드라이버 환경에서 오히려 문제가 생길 수 있어(제안 1의 "다른 구형 드라이버 환경에서 반대로 문제 확인 필요") 검증 없이 전체 기본값을 바꾸는 것은 위험 범위가 넓다. |
| **compute capability를 NVML/CUDA API로 직접 조회한다** | `pynvml`은 `nvmlDeviceGetCudaComputeCapability()`를 제공하지만, 이 저장소는 ADR-0002 이후로 GPU 조회에 `torch`를 끌어들이지 않는 원칙을 지켜왔고 이름 문자열만으로 충분히 식별 가능한 좁은 범위(RTX 50xx)라 API 확장 없이 해결했다. 향후 아키텍처 판별 범위가 넓어지면(예: 데이터센터 카드까지) 재검토 대상이다. |
| **cu126을 후보에서 완전히 빼고 cu128로 대체한다** | ADR-0007이 이미 cu126을 "계속 최신 torch를 받는" 유효한 구간으로 채택했다 — Blackwell이 아닌 대다수 사용자에게는 그대로가 맞다. 특정 아키텍처 예외로만 좁혀 처리하는 편이 영향 범위가 작다. |
| **아무것도 안 하고 안내 문구만 추가한다(이슈 제안 3)** | 근본 원인(아키텍처별 커널 유무)을 이미 알고 있고 판별에 필요한 정보(`gpu_name`)도 이미 조회되고 있어, 실제로 고칠 수 있는데 안내만 하는 것은 회피에 가깝다. |

## 결과

- `env`에 `gpu_name` 필드가 추가된다(`docs/contracts/canary-api.md`, `cli.md`) — `cli.py`·`reverify.py`의 `env` 병합 4곳 모두(ADR-0007과 동일 패턴).
- `--yes`가 실제로 실행하는 `torch_cpu_only_build`의 `fix_command`가 GeForce RTX 50xx에서는 드라이버 major와 무관하게 최소 `cu128`을 받는다.
- 데이터센터 Blackwell(B100/B200/GB200 등)에서는 이번 결정이 적용되지 않는다 — 여전히 ADR-0007의 드라이버 기반 매핑만 적용된다.

## 재확인이 필요한 시점

- PyTorch가 cu126에도 sm_120 커널을 소급 추가하거나(가능성 낮음), cu128 자체가 새로운 torch 버전에서 동결·단종된다.
- NVIDIA가 Blackwell 이후 세대에서 또 다른 신규 compute capability를 내놓아 같은 문제가 반복된다 — 그때는 이 ADR의 패턴(아키텍처 이름 판별 → 최소 태그 강제)을 일반화할지 판단해야 한다.
- 데이터센터/프로 Blackwell 카드 사용자가 실측 문제를 보고하면 이름 판별 범위를 넓힐지 검토한다.
