# loop_lib — Generator↔Detector 공진화 루프 구현 (설계 정본: notes/31-학습-루프-설계.md)

# stdout/stderr 를 UTF-8 로 고정한다. 서버 컨테이너 로케일은 POSIX(LANG 비어 있음),
# 로컬 Windows 콘솔은 cp949 라 둘 다 기본값이면 한글·유니코드 출력에서 죽는다.
# 모든 스크립트가 이 패키지를 import 하므로 여기 한 곳에서 처리한다.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
