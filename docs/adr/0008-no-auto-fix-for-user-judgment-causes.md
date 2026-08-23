# ADR-0008: 사용자 판단이 필요한 세 원인에는 fix_command를 붙이지 않는다

**상태**: Proposed (2026-08-23)
**관련 Issue**: #81, #72, PR #79

## 맥락

PR #79가 `cpu_fallback` 원인을 `env` 기반으로 4갈래(`torch_cpu_only_build` / `torch_cpu_only_build_no_gpu` / `no_nvidia_gpu_or_driver` / `cuda_device_not_visible`)로 나누면서, 그중 `fix_command`가 붙은 것은 NVIDIA GPU가 실제로 보일 때의 `torch_cpu_only_build` 하나뿐이었다. 나머지 세 원인은 안내 문구만 있고 `fix_command`는 `None`으로 남아 있었는데, 이 판단 자체가 어딘가에 결정으로 기록된 적이 없었다 — #72는 "GPU 가려짐은 사용자가 판단할 문제"라며 이 갈래를 스코프에서 뺐을 뿐, 나머지 fix_command 여부까지 결론 내지는 않았다.

#81은 이 세 원인 각각에 "자동 실행해도 안전하고 실제로 문제를 고치는 명령"이 있는지 판단해 달라고 요청했다.

## 결정

**세 원인 모두 `fix_command`를 붙이지 않는다.** 코드는 PR #79 이후로 이미 이 상태였고, 이번 결정은 그 상태를 팀 판단으로 확정하는 것이다.

| cause | 판단 |
|---|---|
| `no_nvidia_gpu_or_driver` | 드라이버 설치·재설치는 AGENTS.md 자동화 등급 D(사람만 실행)에 해당한다. 자동 실행 가능한 명령 자체가 없다. |
| `torch_cpu_only_build_no_gpu` | NVIDIA GPU가 NVML에 아예 안 잡힌 상태다. 이 상태에서 CUDA 빌드 torch(2GB+)를 자동으로 설치해도 GPU가 없으면 결과가 똑같이 cpu로 남는다 — 재검증에서 같은 실패가 반복될 뿐 아무것도 고치지 못한다. |
| `cuda_device_not_visible` | torch도 CUDA 빌드고 NVML도 GPU를 봤는데 프로세스에는 장치가 안 보이는 상태(`CUDA_VISIBLE_DEVICES`로 가려짐, 드라이버/런타임 불일치 등)다. bitsandbytes나 torch 재설치로는 이 상태가 바뀌지 않는다(#72, RTX 4070 Ti 실측) — 원인이 우리 쪽 패키지가 아니라 환경변수·시스템 설정이라 사용자가 직접 판단해야 한다. |

**단, `cuda_device_not_visible`만 진단 정보를 하나 추가한다**: 안내 문구에 현재 `CUDA_VISIBLE_DEVICES` 값을 그대로 보여준다(`env.cuda_visible_devices`, `src/preflight/canary/worker.py`가 `os.environ.get()`으로 채운다). 이건 `fix_command`가 아니다 — 아무것도 자동 실행하지 않고, 사용자가 원인을 좁히는 데 필요한 사실 하나를 화면에 얹을 뿐이다.

## 대안

| | 왜 안 택했나 |
|---|---|
| **`no_nvidia_gpu_or_driver`에 드라이버 설치 명령을 붙인다** | 드라이버 설치는 시스템 전역에 영향을 주고 재부팅이 필요할 수 있는 등 되돌리기 어렵다 — AGENTS.md가 이미 Grade D로 명시한 영역이라 재론의 대상이 아니다. |
| **`torch_cpu_only_build_no_gpu`에도 `torch_cpu_only_build`와 같은 재설치 명령을 붙인다** | "GPU가 있다면"이라는 가정이 성립하지 않는 상태에서 실행하는 것이므로, `--yes`가 아무것도 고치지 못하는 명령을 실제로 실행하게 된다(#51·#55·#72와 같은 유형의 버그). |
| **`cuda_device_not_visible`에 `CUDA_VISIBLE_DEVICES` 값을 자동으로 지우거나 재설정한다** | 사용자가 의도적으로 설정했을 수 있는 환경변수를 도구가 임의로 바꾸는 것이라 부작용 범위를 예측할 수 없다. 진단 정보 제공으로 충분하다. |
| **세 원인 다 그대로 두고 문서화하지 않는다** | #81이 지적한 대로 "결론 안 낸 사안"으로 남아 다음에 같은 질문이 반복된다. |

## 결과

- `--yes`가 실제로 pip 명령을 실행하는 원인은 여전히 `bnb_not_compiled_with_cuda`·`torch_cpu_only_build` 둘뿐이다 — 이번 결정으로 늘어나지 않는다.
- `cuda_device_not_visible`의 `env`에 `cuda_visible_devices` 필드가 추가된다(`docs/contracts/canary-api.md`).
- 세 원인 모두 향후 실측으로 "실제로 안전하게 고칠 수 있는 명령이 있다"는 근거가 나오기 전까지는 이 결정을 유지한다.
