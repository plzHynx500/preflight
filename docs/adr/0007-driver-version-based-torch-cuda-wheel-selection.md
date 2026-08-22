# ADR-0007: torch CUDA 재설치 휠은 드라이버 major 브랜치 번호로만 고른다

**상태**: Accepted (2026-08-23)
**관련 Issue**: #82

## 맥락

`torch_cpu_only_build` 원인의 `fix_command`는 PR #79까지 `cu124` 인덱스로 고정돼 있었다. `env`에 드라이버 버전이 없어 고를 근거가 없었기 때문이다(#82가 지적).

#82를 구현하며 `env.gpu_driver_version`을 먼저 추가한 뒤(같은 PR), 실제로 드라이버 버전 → CUDA 휠을 매핑하려고 두 가지를 조사했다.

**① NVIDIA CUDA Toolkit Release Notes의 "Toolkit Driver Version" 표**(2026-08 확인) — Linux와 Windows가 서로 다른 최소 드라이버 버전을 쓴다.

| CUDA | Linux 최소 | Windows 최소 |
|---|---|---|
| 12.6 GA | 560.28.03 | 560.76 |
| 12.8 GA | 570.26 | 570.65 |
| 12.9 GA | 575.51.03 | 576.02 |
| 13.0 GA | 580.65.06 | 공식 미공개 (13.0부터 Windows 드라이버가 툴킷 패키지에서 분리됨) |

`env.gpu_driver_version`은 NVML이 OS 원본 문자열을 그대로 준다 — Linux는 보통 세 자리(`"560.28.03"`), Windows는 두 자리(`"595.79"`, 이 저장소 개발 머신 RTX 5070 Laptop GPU 실측)다. 표의 patch 자리 값도 OS마다 다르다.

**② `download.pytorch.org/whl/<tag>/torch/`를 직접 조회**(2026-08, 실제 인덱스) — PyTorch가 배포 중인 CUDA 태그별 최신 torch 버전.

| 태그 | 최신 torch |
|---|---|
| cu118 | 2.7.1 |
| cu121 | 2.5.1 |
| cu124 | 2.6.0 |
| **cu126** | **2.13.0** |
| cu128 | 2.11.0 |
| cu129 | 2.9.0 |
| **cu130** | **2.13.0** |

cu126·cu130만 최신 torch를 계속 받고, 나머지 태그는 특정 버전에서 **동결**돼 있다.

## 결정

**드라이버 버전 문자열의 major 브랜치 번호(첫 `.` 앞자리)만 보고, cu126·cu130 두 구간만 매핑에 쓴다. 그 외에는 기존 기본값 `cu124`로 떨어진다.**

```
driver major >= 580  →  cu130   (CUDA 13.0 GA, Linux 최소 580.65.06)
driver major >= 560  →  cu126   (CUDA 12.6 GA, Linux 560.28.03 / Windows 560.76)
그 외 / 파싱 실패 / 값 없음  →  cu124  (기존 고정값 그대로)
```

구현: `src/preflight/fix/executor.py`의 `_torch_cuda_tag_for_driver()` · `_TORCH_CUDA_TAG_BY_MIN_DRIVER_MAJOR`. `suggest_fix()`가 `torch_cpu_only_build`일 때만 `env.gpu_driver_version`을 읽어 동적으로 `fix_argv`를 만든다.

**major 번호만 보는 이유**: 위 표에서 Linux·Windows 최소값이 patch 단위로는 다르지만 major 브랜치 번호(560/570/580)는 같다. patch까지 맞추려면 OS를 먼저 판별해야 하는데, `env`에는 OS 정보가 없고(canary-api.md `env` 절 범위 밖) 넣으려면 계약을 또 넓혀야 한다. major 번호만 비교하면 OS 구분 없이 안전한 근사가 된다 — 경계에 걸친 소수의 드라이버(예: Linux 560.10처럼 major는 560이지만 실제 12.6 patch 요건에는 못 미치는 경우)는 cu126을 시도해 pip 쪽에서 실패할 수 있지만, 실패하더라도 재확인이 그대로 FAIL로 남을 뿐 기존 cu124 고정 시절보다 나빠지지 않는다.

**cu121·cu124·cu128·cu129를 후보에서 뺀 이유**: "드라이버가 지원하는 가장 높은 CUDA"를 그대로 고르면, 동결된 태그를 골라 지금 받을 수 있는 최신 torch(2.13.0)보다 오래된 버전을 설치하는 역효과가 난다. 목표는 "드라이버에 맞으면서 최신인 torch"이지 "드라이버가 지원하는 가장 높은 CUDA 번호" 자체가 아니다.

**cu124 기본값을 유지한 이유**: 이미 검증된 안전한 폴백이고(PR #79부터 실사용), 이슈 원안이 "매핑에 없으면 지금처럼 기본값"이라고 명시했다.

## 대안

| | 왜 안 택했나 |
|---|---|
| **ⓐ patch 단위까지 정확히 매핑 + OS 판별** | `env`에 OS 필드를 추가해야 하는 별도 계약 변경이고, CUDA 13.0의 Windows 최소값이 아직 공식 미공개라 정확한 patch 표 자체를 못 만든다. 얻는 정밀도 대비 비용이 크다 |
| **ⓑ pypi.org/pytorch 메타데이터를 런타임에 조회해 최신 사용 가능 태그를 그때그때 판단** | `--yes` 경로가 네트워크 호출에 의존하게 되고 실패 시 폴백이 또 필요해진다. 오프라인/사내망 환경(이 도구의 주 사용 시나리오 중 하나)에서 깨진다 |
| **ⓒ 지금처럼 cu124 고정 유지 (아무것도 안 함)** | #82가 이미 지적한 문제(최신 드라이버에서 오래된 torch로 고정)가 그대로 남는다 |

## 결과

- `--yes`가 실제로 실행하는 유일한 pip 명령(`torch_cpu_only_build`)이 드라이버에 따라 cu124/cu126/cu130 셋 중 하나로 갈린다.
- `torch_cpu_only_build_no_gpu`는 그대로 cu124 안내 문구를 쓴다 — 이 갈래는 애초에 `gpu_free_mb`가 없어(GPU 자체가 안 보임) `gpu_driver_version`도 없으므로 매핑에 들어갈 정보가 없다.

## 재확인이 필요한 시점 — 표가 어긋나는 조건

이 표는 **정적 스냅샷**이다. 아래 중 하나가 생기면 다시 조사해야 한다.

- PyTorch가 새 CUDA 태그를 열거나(예: cu131), 기존 cu126/cu130 중 하나를 동결한다.
- NVIDIA가 새 CUDA GA를 낸다(다음은 13.1+로 예상, Windows 드라이버가 툴킷과 완전히 분리된 이후라 최소 버전 확인 방법 자체가 바뀔 수 있다).
- CUDA 13.0의 공식 Windows 최소 드라이버 버전이 뒤늦게 공개된다 — 지금 `580` 임계값은 Linux 표만 근거로 삼은 값이라 정정이 필요할 수 있다.
