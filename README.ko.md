# GhostRelay OpenClaw Patch

GhostRelay OpenClaw Patch는 Ghost Chat/Companion UX를 OpenClaw와 통합하기 위한 사이드카 패치 패키지입니다.

## What / Why

GhostRelay는 OpenClaw 주변에 데스크톱 제어면(Control Plane)을 제공하여:

- 하나의 UI에서 채팅/컴패니언 워크플로를 운영하고
- 프로필별 워크스페이스/세션을 분리하며
- 원시 설정 파일 직접 편집 없이 게이트웨이/모델 흐름을 관리할 수 있게 합니다.

핵심 포지션은 “대체”가 아니라 “패치형 사이드카”입니다.

## 핵심 기능

- Launcher에서 게이트웨이 제어(`start/stop/restart`) + 트레이 관리
- Ghost Chat의 프로필 기반 모드(`auto`, `chat`, `agent`)
- Companion 채팅의 브레인 준비/메모리 동기화(`BRAIN_SUMMARY.md`)
- Ollama 모델 관리(목록/등록/제거)
- Persona/Memory/Session/Skills 사이드패널 편집
- allowlist 기반 공개 번들 생성 스크립트

## 통합 구조

```text
project-root/
  ghostchat/
  openclaw-main/
```

- `ghostchat`: 사이드카 앱
- `openclaw-main`: 외부 의존성
- `ghostchat`는 별도 저장소/패키지로 배포 가능

## 설치

```powershell
pip install -r requirements.txt
```

대시보드 기능 사용 시 Node.js 18+ 필요.

## 실행

```powershell
python launcher.py
```

또는:

```powershell
start.bat
```

- 기본: 최소화 + `pythonw`
- 콘솔 모드: `start.bat --console`

## 상태

- 버전: `0.4.0-beta.1` (`ghostchat/VERSION`)
- 단계: `Beta`
- 기준일: `2026-02-08`

### 검증됨

- `python -m py_compile`로 핵심 파이썬 파일 컴파일 검증
- 공개 번들 스크립트 실행 성공
- 번들에 `LICENSE`/`NOTICE` 포함, 개인 런타임 데이터 제외 확인
- GitHub 저장소 부트스트랩/푸시 경로 검증

### 아직 미완성

- 완전 자동 멀티에이전트 오케스트레이션
- 장시간 세션 프리징/지연에 대한 광범위 안정화
- UI 전 경로 E2E 자동 테스트

## 협업 모델

권장 오픈소스 흐름:

- `main` 보호
- 브랜치/포크 기반 자유 기여
- PR 머지 방식 운영

## Privacy & Public Repo Policy

절대 커밋 금지:

- `settings.json`, `logs/`
- `workspace/`, `profiles/`, `memory/`
- `.env*`, `*credentials*.json`, `*service-account*.json`, `*.pem`, `*.key`
- 모델 파일(`*.gguf`, `*.safetensors`, `*.onnx`)

보호 장치:

- `.gitignore`
- 공개 allowlist(`release_strategy/PUBLISH_ALLOWLIST.txt`)
- (선택) pre-commit blocklist

## 공개 번들 생성

```powershell
powershell -ExecutionPolicy Bypass -File .\ghostchat\release_strategy\create_public_bundle.ps1
```

출력:

- `ghostchat-public-bundle/ghostchat`

## 버전 정책 (임시)

- 형식: `MAJOR.MINOR.PATCH[-label]`
- 기준 파일:
  - `ghostchat/VERSION`
  - `ghostchat/CHANGELOG.md`

## License & Attribution

- 라이선스: `MIT` (`ghostchat/LICENSE`)
- 저작자: Yun Jin Gyu (`akrnwkql@gmail.com`)
- 재배포 시 `LICENSE`/`NOTICE` 유지
