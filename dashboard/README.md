# OpenClaw Skills Dashboard

Portable, copyable dashboard to view and toggle OpenClaw skills.

## Quick Start

**Windows**: `start.bat` 더블클릭 또는 실행

```powershell
.\start.bat
```

브라우저가 자동으로 열립니다: `http://127.0.0.1:5177`

## Manual Start

```powershell
node server.mjs
```

Open: `http://127.0.0.1:5177`

## 배포 요구사항

### 필수 요구사항

- **Node.js**: Node.js가 설치되어 있어야 합니다 (PATH에 등록되어 있으면 자동 탐색)
- **OpenClaw**: `openclaw-main` 폴더가 아래 둘 중 하나 위치에 있어야 합니다 (자동 탐색)
  - `ghostchat/openclaw-main`
  - `ghostchat` 상위 폴더의 `openclaw-main`
- **설정 파일**: OpenClaw 설정 파일 (`~/.openclaw/openclaw.json`)이 있어야 합니다

### 배포 방법

**대시보드 폴더만 복제하면 안 됩니다!**

필요한 것:
1. `ghostchat` 폴더 (내부에 `dashboard` 포함)
2. `openclaw-main` 폴더 (위치: `ghostchat` 내부 또는 상위 폴더)

배치:
```
프로젝트/
  ├── ghostchat/
  │   ├── dashboard/      ← 대시보드
  │   └── openclaw-main/  ← 선택(내부 배치)
  └── openclaw-main/      ← 선택(상위 배치)
```

그 다음 `ghostchat/dashboard/start.bat` 실행

## Environment Variables

- `OPENCLAW_WORKSPACE`: 프로젝트 루트 경로 (자동 탐색됨)
- `OPENCLAW_CLI`: OpenClaw CLI 경로 (자동 탐색됨)
- `OPENCLAW_CLI_ARGS`: CLI 명령 오버라이드 (JSON 배열)
- `HOST`, `PORT`: 서버 주소 및 포트 (기본값: `127.0.0.1:5177`)

## 자격상태 vs 활성화

### 자격있음 (Eligible) vs 활성화 (Enabled)

**자격있음 (Eligible)**:
- 스킬이 실행 가능한 상태인지 (필수 요구사항 충족 여부)
- 시스템이 자동으로 판단
- 사용자가 직접 켜고 끌 수 없음
- 요구사항을 충족시키면 자동으로 "자격있음"이 됨

**활성화 (Enabled)**:
- 사용자가 스킬을 사용할지 말지 결정
- 대시보드에서 토글 스위치로 켜고 끔
- 설정 파일에 저장됨

### 자격있음을 만드는 방법

필수 요구사항을 충족시키면 자동으로 "자격있음"이 됩니다:

- 프로그램 설치 (예: `git`, `python`)
- 환경변수 설정 (예: `API_KEY`)
- 설정 파일 생성
- OS 호환성 확인

Missing 컬럼에서 어떤 요구사항이 부족한지 확인할 수 있습니다.

## Notes

- Toggling a skill updates `skills.entries.<skillKey>.enabled` and requires a gateway restart.
- If the CLI is not on PATH, set `OPENCLAW_CLI` or `OPENCLAW_CLI_ARGS`.
- **자격있음은 사용자가 직접 변경할 수 없습니다.** 요구사항을 충족시키면 자동으로 변경됩니다.
