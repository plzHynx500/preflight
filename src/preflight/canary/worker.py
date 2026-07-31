"""subprocess로 격리 실행되는 canary 본체 (forward+backward+optimizer.step()+측정).

engine.run_canary_check()이 `python -m preflight.canary.worker`로 이 모듈을
별도 프로세스에서 기동한다. import 크래시·OOM이 나도 부모 프로세스는 죽지
않아야 한다 (docs/adr/0002-subprocess-isolation-for-canary.md 참고).
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
