# Preflight

> 파인튜닝을 시작하기 전, 지금 환경이 실제로 준비됐는지 확인합니다.

*오픈소스 · 파인튜닝 환경 진단 도구*

```
pip install preflight-gpu
preflight check
```

---

## 요구 환경

| | |
|---|---|
| GPU | **NVIDIA GPU와 CUDA 드라이버** |
| OS | Linux, Windows (네이티브) |
| Python | 3.9 이상 |

**macOS(Apple Silicon)와 AMD ROCm은 아직 대상이 아닙니다.** 설치는 되지만 CUDA 경로만 보기 때문에 쓸 만한 답을 주지 못합니다. GPU가 아예 없는 환경에서도 "없다"는 것까지는 알려주지만, 그 뒤로 할 수 있는 일이 없습니다.

---

## 왜 만들었나

QLoRA 같은 경량화 기법 덕분에 개인 GPU 한 장으로도 LLM 파인튜닝이 현실적인 선택지가 됐습니다. 하지만 드라이버·CUDA·PyTorch·학습 라이브러리로 이어지는 버전 호환성과 VRAM 산정은 여전히 까다롭고, 무엇보다 **에러 없이 조용히 실패하는 경우**가 흔합니다.

**조용한 실패.** GPU가 꽂혀 있어도 연산이 CPU로 도는 일이 있습니다. 느릴 뿐 에러가 없어서 몇 시간 뒤에야 눈치챕니다. `torch.cuda.is_available()`이 True이고 버전도 다 맞는데 조용히 CPU로 폴백돼 RAM만 먹던 사례를 직접 겪었습니다.

**VRAM 오판.** 학습에는 가중치·그래디언트·옵티마이저 상태·활성값이 동시에 올라갑니다. 앞의 셋은 모델 구조만 알면 계산되지만, 활성값과 그로 인한 allocator 단편화는 돌려봐야 압니다.

**버전 체인.** 드라이버·CUDA·PyTorch·학습 라이브러리가 서로 물려 있어 하나만 어긋나도 학습이 시작조차 안 됩니다. `pip check`가 통과시키는 조합에서도 그렇습니다.

---

## 어떻게 확인하나

버전 문자열을 읽고 "괜찮아 보인다"고 말하지 않습니다. 실제로 4bit 레이어를 만들고 LoRA 어댑터를 붙여 **학습 스텝을 한 번 돌려봅니다.**

가중치나 데이터셋은 받지 않습니다. GPU 메모리 점유량은 텐서의 값이 아니라 shape와 dtype이 정하므로, 구조만 같은 랜덤 모델로도 같은 답이 나옵니다. 모델 config만 조회하고(수 KB) 같은 모양의 모델을 만들어 돌립니다.

canary는 **별도 프로세스에서** 돕니다. 진단이 필요한 상황이 곧 `import torch`가 죽는 상황이라, 같은 프로세스에서 돌리면 도구가 함께 죽어 아무것도 알려줄 수 없습니다.

```
드라이버·CUDA  →  Python·학습 라이브러리  →  [ Preflight ]  →  실제 학습 실행
  (OS 레벨)          (환경 구성)              (여기서 확인)
```

### 무엇을 기준으로 재는가

이 진단은 **QLoRA(4bit 양자화 + LoRA 어댑터) + AdamW**를 기준으로 잽니다. 기본 체크는 batch=1 · seq=8 고정입니다.

그래서 4bit을 쓸 생각이 없는 사용자에게는 `⚠ 4bit 사용 불가` 같은 줄이 나올 수 있습니다. **bf16 + LoRA만 쓴다면 그 줄은 무시해도 됩니다.** 반대로 `device=cpu`나 VRAM 관련 판정은 어떤 학습 방식이든 그대로 해당됩니다.

기준을 바꾸는 옵션(`--quantization`, `--optimizer` 등)은 아직 없습니다. 다음 버전에서 다룰 예정입니다.

**진단하는 것은 환경입니다.** 드라이버·CUDA·torch 빌드·라이브러리 설치·VRAM까지가 대상이고, **학습 코드 자체의 오류나 라이브러리 API 변경은 잡지 않습니다.** preflight를 통과했는데 `TypeError: SFTConfig.__init__() got an unexpected keyword argument` 같은 것으로 죽는다면, 그건 환경이 아니라 코드 쪽 문제입니다.

---

## 무엇을 하나

**확인 — Canary Check.** 작은 텐서 연산 하나로 device 배치, GPU 메모리 이동, CPU 대비 실행 속도를 함께 재서 조용한 실패를 잡습니다.

**확인 — VRAM 실측.** `--model`을 주면 가중치 다운로드 없이 그 모델 구조로 canary를 구성해 메모리 사용량을 잽니다. 계산이 아니라 실측이라 활성값과 단편화가 반영됩니다.

**수정 + 재확인.** 문제에 맞는 명령을 제시하고, `--yes`를 주면 실행한 뒤 다시 재서 실제로 고쳐졌는지 보여줍니다.

---

## 설치

```
pip install preflight-gpu
```

설치되는 의존성은 `typer`·`rich`·`nvidia-ml-py` 셋뿐입니다. **torch·transformers·bitsandbytes는 일부러 넣지 않았습니다.** 사용자 환경에 이미 깔린 버전을 그대로 불러와 검증하는 것이 목적이라, 진단 도구의 설치가 진단 대상을 덮어쓰면 안 되기 때문입니다.

설치 후 `preflight` 명령을 못 찾으면(venv를 활성화하지 않았거나 `--user`로 설치한 경우 Windows에서 흔합니다) 모듈로 실행하면 됩니다.

```
python -m preflight check
```

> ⚠️ **패키지 이름은 `preflight-gpu`입니다. 실행 명령만 `preflight`입니다.**
>
> PyPI의 `preflight`는 **무관한 다른 패키지**(웹사이트 배포 점검 도구, BSD)입니다. `pip install preflight`는 **에러 없이 성공하고** 그 도구가 깔립니다 — 그쪽도 `preflight`라는 콘솔 명령을 설치하기 때문에, 같은 환경에 둘 다 있으면 나중에 설치한 쪽이 명령을 덮어씁니다.
>
> 헷갈릴 때는 [pypi.org/project/preflight-gpu](https://pypi.org/project/preflight-gpu/)에서 확인하세요.

---

## 언제 쓰나요 — 학습 명령어 실행 직전

`git commit` 전에 `git status`를 확인하듯, 학습을 시작하기 전 한 줄이면 됩니다. 설치 직후, 드라이버 업데이트 직후, 팀 온보딩이나 CI 스크립트에 넣어두기를 권합니다.

아래는 RTX 4070 Ti에서 실제로 나온 출력입니다.

```
$ preflight check

✔ Canary 연산 실행    device=cuda · 메모리 이동 확인됨
✔ 실행 시간 2ms    CPU 대비 5배 (정상 범위)
ℹ 4bit 레이어 폴백    nn.Linear로 대체 실행됨
⚠ 4bit 사용 불가    bitsandbytes가 없어 QLoRA(4bit)로 학습할 수 없습니다
    이 상태로 QLoRA 학습을 시작하면 ImportError로 즉시 종료됩니다.

FIX: C:\venv\Scripts\python.exe -m pip install -U bitsandbytes>=0.46.1
재확인: preflight check --yes

4개 항목 확인 · 1개 문제 발견 · 소요 시간 4초
```

제안하는 명령은 PATH에서 찾은 `pip`이 아니라 **지금 preflight를 실행 중인 파이썬**을 가리킵니다. venv를 활성화하지 않았거나 pipx·전역 설치로 쓰는 환경에서, 진단한 곳과 다른 환경에 설치되는 것을 막기 위해서입니다.

`--model`을 주면 그 모델을 이 GPU에 올렸을 때가 나옵니다.

```
$ preflight check --model google/gemma-4-12B-it

모델 체크: google/gemma-4-12B-it
✔ Canary 연산 실행    device=cuda · 메모리 이동 확인됨
✔ VRAM 실측    9.7GB / 4.6GB 가용 (총 12GB)
✔ 목표 배치 크기 적합    batch=1, seq=8 기준
⚠ VRAM 여유    가용 VRAM의 208% 소모 — 실제 학습 시 OOM 위험 높음
```

### CI 연동과 종료 코드

CI 파이프라인이나 스크립트에서 결과를 분기할 수 있도록 종료 코드를 규격화했습니다. `--yes`로 재확인까지 했다면 재확인 결과로 교체된 전체를 다시 집계한 값이 기준입니다. 실행할 명령이 없었거나 명령이 실패해 재확인을 못 했으면 1차 판정이 기준입니다.

| 종료 코드 | 판정 | 설명 |
|---|---|---|
| `0` | **PASS** | 모든 판정 항목이 정상 |
| `1` | **FAIL** | 최종 판정에 FAIL이 하나라도 포함된 경우 |
| `2` | **WARN** | FAIL 없이 WARN만 포함된 경우 |

`--json`을 주면 같은 내용이 구조화된 형태로 나옵니다.

> **주의** — GitHub Actions·Jenkins 등 대부분의 CI 도구는 `0`이 아닌 모든 종료 코드를 파이프라인 실패로 봅니다. WARN(`2`)에서 중단하지 않으려면 셸에서 분기해야 합니다.
> ```bash
> preflight check || [ $? -eq 2 ]
> ```

---

## 누가 쓰면 좋은가

파인튜닝 도구 설치부터 막혀 학습 화면조차 못 본 사람, "문제가 있을 수도 있다"가 아니라 확정된 답이 필요한 사람, 드라이버를 갱신하거나 GPU를 바꾼 뒤 매번 다시 확인하고 싶은 사람입니다.

학습 도구를 무엇으로 고르든 그 앞단은 똑같이 드라이버·CUDA·PyTorch입니다. 그 공통 구간을 먼저 확인해서, 몇 시간짜리 학습이 환경 문제로 도중에 죽는 일을 줄이는 것이 목표입니다.

---

## 프로젝트 정보

| | |
|---|---|
| 라이선스 | MIT — 재배포·임베딩 제약 없음 |
| 지원 플랫폼 | Linux, Windows (네이티브) · NVIDIA GPU 필요 |
| Python | 3.9 이상 |
| 설치 | `pip install preflight-gpu` (실행 명령은 `preflight`) |
| PyPI | [pypi.org/project/preflight-gpu](https://pypi.org/project/preflight-gpu/) |
| 실행 | `preflight check` — 명령을 못 찾으면 `python -m preflight check` |
| 종료 코드 | 0 = PASS · 1 = FAIL · 2 = WARN |

---

*기여, 이슈 제보, 피드백을 환영합니다. 자세한 설계 배경은 저장소의 `docs/`를 참고해 주세요.*
