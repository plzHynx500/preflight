"""`python -m preflight` 진입점.

콘솔 스크립트(`preflight`)와 같은 일을 한다. 이 파일이 있어야 하는 이유는 두 가지다.

1. **Windows에서 `Scripts/`가 PATH에 없는 경우가 흔하다** — venv를 활성화하지 않았거나
   `pip install --user`로 깔면 `preflight` 명령을 못 찾는다. 그때의 표준 폴백이
   `python -m <패키지>`인데, `__main__.py`가 없으면 "cannot be directly executed"로
   막힌다(#64).
2. **어느 파이썬으로 도는지 명시할 수 있다** — 진단 도구는 "지금 이 파이썬 환경"을
   보는 게 목적이라, `<그 venv>/python.exe -m preflight`처럼 인터프리터를 직접
   지정하는 호출이 의미가 있다. `--yes`의 수정 명령이 `sys.executable`을 쓰는 것과
   같은 이유다.
"""

from preflight.cli import app

if __name__ == "__main__":
    app()
