# GhostRelay

OpenClaw 옆에 붙여 쓰는 데스크톱 채팅/런처 패치 패키지입니다.

## 현재 상태

- 버전: `0.4.0-beta.1` (`ghostchat/VERSION`)
- 단계: `Beta` (기능 확장 중)
- 기준일: `2026-02-08`

## 검증 상태 (중요)

### ✅ 실행 검증 완료

- `python -m py_compile`로 핵심 파이썬 파일 문법 검증 완료
- `release_strategy/create_public_bundle.ps1` 실행 및 번들 생성 확인 완료
- 공개 번들에 `LICENSE`/`NOTICE`/`.gitignore` 포함, 민감 파일 제외 동작 확인 완료
- GitHub Private 저장소 생성 + 첫 push 경로 검증 완료

### 🟡 코드 구현됨 (수동 UI 검증 필요)

- Launcher에서 Gateway 시작/중지/재시작 버튼
- Ghost Chat 메인 창 + Companion 창 + 트레이 관리
- 프로필별 워크스페이스 분리/저장
- Ollama 모델 관리(목록/등록/제거) 다이얼로그
- Companion 브레인 준비/메모리 동기화 및 `BRAIN_SUMMARY.md` 저장 경로

### 🔴 아직 미완성/안정화 필요

- 완전 자동 멀티에이전트 오케스트레이션(무인 파이프라인)
- 장시간 실행 시 프리징/지연에 대한 광범위 회귀 테스트
- 모든 UI 플로우의 E2E 자동 테스트

## 프로젝트 구조

```text
ghostchat/
  launcher.py
  ghost_chat.py
  ghost_chat_companion.py
  settings_store.py
  runtime_paths.py
  local_logger.py
  start.bat
  start_gateway.bat
  requirements.txt
  package.json
  VERSION
  CHANGELOG.md
  LICENSE
  NOTICE
  dashboard/
  release_strategy/
```

## 설치

### Python

```powershell
pip install -r requirements.txt
```

### Node.js

대시보드 사용 시 Node.js 18+ 필요.

## 실행

### Launcher 실행

```powershell
python launcher.py
```

또는

```powershell
start.bat
```

- 기본: 최소화 + `pythonw` 실행
- 콘솔 모드: `start.bat --console`

### Gateway 스크립트 직접 실행

```powershell
start_gateway.bat
```

## 패치형 통합 방식

GhostRelay는 OpenClaw를 대체하지 않고 옆에 붙는 도구입니다.

```text
project-root/
  ghostchat/
  openclaw-main/
```

- `openclaw-main`은 외부 의존성으로 유지
- `ghostchat`만 별도 저장소로 배포 가능

## 현재 확인된 주요 기능

### Launcher

- 스킬 대시보드 열기
- Ghost Chat 창 생성/관리
- Companion Chat 창 생성/관리
- Gateway 상태 표시 및 제어 버튼
- Ollama 모델 관리 다이얼로그

### Ghost Chat

- 프로필 선택/적용
- 대화 모드: `auto` / `chat` / `agent`
- 좌/우 패널에서 Persona/Memory/Session/Skills 편집
- 로컬 관리 명령(`/session`, `/skill`) 처리 경로

### Companion

- 브레인 프로필 선택
- 시작 시 브레인 준비 요청
- 메모리 동기화 요청
- `workspace/.../memory/BRAIN_SUMMARY.md` 생성/갱신 경로

## 알려진 제약

- Ollama 모델이 tool-calling 미지원이면 `agent` 모드가 실패할 수 있음
- 프로필 변경 직후 일부 설정은 Gateway 재시작 후 안정적으로 반영됨
- UI 응답성은 환경/모델/컨텍스트 크기에 따라 차이가 큼

## 공개 저장소 정책

### Attribution (출처 표기)

- 라이선스: `MIT`
- 저작권자: Yun Jin Gyu (`akrnwkql@gmail.com`)
- 재배포/수정 배포 시 `LICENSE`, `NOTICE` 유지

### Privacy (개인 데이터 비공개)

절대 커밋 금지:

- `settings.json`, `logs/`
- `workspace/`, `profiles/`, `memory/`
- `.env*`, `*credentials*.json`, `*service-account*.json`, `*.pem`, `*.key`
- `*.gguf`, `*.safetensors`, `*.onnx`

기본 보호:

- `.gitignore` 반영
- 공개 번들 allowlist 반영
- (선택) `pre-commit` 훅 지원

## 공개 번들 생성

```powershell
powershell -ExecutionPolicy Bypass -File .\ghostchat\release_strategy\create_public_bundle.ps1
```

결과:

- `ghostchat-public-bundle/ghostchat`

## 버전 관리 (임시 정책)

- 형식: `MAJOR.MINOR.PATCH[-label]`
- 현재 채널: `beta`
- 기준 파일:
  - 현재 버전: `ghostchat/VERSION`
  - 변경 이력: `ghostchat/CHANGELOG.md`

릴리즈 규칙(임시):

- 기능 추가: `MINOR` 증가
- 버그 수정: `PATCH` 증가
- 큰 구조 변경: `MAJOR` 증가

## 라이선스

- `MIT` (`ghostchat/LICENSE`)
- 누구나 사용/수정/재배포/상업적 사용 가능
- 단, 라이선스/저작권 고지는 유지해야 함
