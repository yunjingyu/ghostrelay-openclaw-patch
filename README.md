# GhostRelay

BRAIN 전용 채팅 인터페이스 및 툴 모음집 (배포 가능한 독립 패키지)

## 구조

```
ghostchat/
  ├── launcher.py          # 시스템 트레이 + 툴 모음집
  ├── ghost_chat.py        # BRAIN 전용 고스트 챗
  ├── start_gateway.bat    # 내장 OpenClaw 게이트웨이 시작 스크립트
  ├── runtime_paths.py     # 경로 자동 탐색(독립 배포용)
  ├── local_logger.py      # 내장 로거
  ├── dashboard/           # 스킬 대시보드 (통합됨)
  │   ├── server.mjs      # 대시보드 서버
  │   ├── public/         # UI 파일들
  │   │   ├── index.html
  │   │   ├── app.js
  │   │   └── styles.css
  │   ├── lib/            # 유틸리티
  │   │   └── skills-clusters.mjs
  │   └── start.bat       # 대시보드 실행 스크립트
  ├── requirements.txt     # Python 의존성 (PySide6)
  ├── package.json         # Node.js 의존성 (선택적)
  ├── start.bat            # 메인 실행 스크립트
  └── README.md
```

## 기능

### Ghost Chat
- OpenClaw Gateway 상태 확인 후 즉시 연결 (자동 시작 없음)
- BRAIN과 실시간 채팅
- WebChat WS 우선 + HTTP(OpenAI 호환) 자동 폴백
- Ollama 모델이 tools 미지원일 때 provider별 무툴 정책 자동 적용 후 즉시 재시도
- 프로필별 `대화 모드` 지원: 자동(모델 감지) / 대화 전용(도구 OFF) / 에이전트(도구 ON)
- Ollama 모델 선택 시 tool-calling 지원 여부 선제 감지 + 미지원 시 자동 대화 전용 전환
- 대화 전용 + Ollama 선택 시 OpenClaw agent WS 대신 Ollama OpenAI endpoint로 직접 대화 (tools 강제 비활성)
- Thinking 레벨 선택 (`off/minimal/low/medium/high`)
- 응답 중 상태 표시 (`생각중... + 경과초`) + 스트리밍 미리보기
- 사고접기 ON 시 중간 사고/도구 출력은 접혀 저장되고, 응답 메시지의 링크로 열람 가능
- 게이트웨이 실행 중이면 즉시 연결 (불필요한 재시작/대기 최소화)
- 기본 `sessionKey`를 런타임 키로 분리해 과거 누적 컨텍스트 병목 완화
- 창 모서리 드래그 리사이즈 지원 (프레임리스 창)
- Vertex 429(Resource Exhausted) 오류 즉시 감지 및 빠른 안내 (불필요한 장시간 재시도 방지)
- 투명도 조절
- 드래그/드롭
- 시스템 트레이 지원
- `옵션` 버튼으로 좌/우 편집 패널 열기/닫기
- 좌측 Persona 패널: `workspace/AGENTS.md`, `SOUL.md`, `IDENTITY.md`, `USER.md`, `TOOLS.md`, `HEARTBEAT.md` 편집
- 우측 Control 패널:
  - Memory: `workspace/MEMORY.md` + `workspace/memory/YYYY-MM-DD.md` 편집
  - Persona/Memory 파일 외부 변경 자동 감지 + 자동 새로고침 (미저장 편집 중이면 보류)
  - Memory 설정 토글: `memorySearch.enabled`, `sources`, `experimental.sessionMemory`, `sync.onSessionStart`
  - Session 설정: `session.dmScope`, `session.identityLinks`(테이블 폼), GhostChat `sessionKey`
  - Skills 설정: 목록 조회, 수동 활성/비활성, ClawHub 설치
- 헤더에서 창별 프로필 선택 + 설정창 진입
- 창별 프로필별 `agentId`/`workspace`/`model`/`sessionKey` 분리
- 비-main 프로필은 `workspace/profiles/<profile>` 격리 워크스페이스로 자동 보정
- 프로필 워크스페이스에 persona/memory 기본 파일 자동 스캐폴딩
- Companion 채팅 모드(대화 전용): 브레인 프로필 선택 + 자동 준비(세션 유지) 브리지 지원
- Companion 채팅에서 브레인 준비/메모리 동기화 시작 시 선택된 브레인 프로필의 브레인 채팅창 자동 표시(모니터링용)
- Companion 채팅의 브레인 준비/메모리 동기화 이벤트를 브레인 채팅창에 실시간 로그로 미러링
- Companion 우측 패널은 메모리 중심으로 단순화(일일 메모 / MEMORY.md / BRAIN_SUMMARY.md)
- Companion 브레인 요약은 툴-미지원 모델 오류 시 직접 Ollama chat 경로로 자동 fallback 후 `BRAIN_SUMMARY.md` 저장

### Launcher
- 스킬 대시보드 실행
- 브레인 채팅 실행 (클릭할 때마다 새 창)
- Companion 채팅 실행 (클릭할 때마다 새 창)
- 게이트웨이 상태 표시 (빨강/초록)
- 게이트웨이 시작/중지/재시작 버튼
- Ollama 모델 관리 다이얼로그(gguf 선택 → Modelfile 생성 → `ollama create` 등록 → 목록 새로고침 → 선택 제거)
- 콘솔 트레이에서 Ghost Chat 창 목록 동적 관리 (열기/설정/숨기기/종료)
- 버튼 클릭 시 자동 트레이 최소화
- 트레이에서 숨김/복귀/종료 가능
- 설정 저장/적용은 백그라운드 적용(프리징 완화)

### Skills Dashboard
- OpenClaw 스킬 목록 및 상태 확인
- 스킬 활성화/비활성화
- 게이트웨이 상태 확인 및 재시작
- 자격 획득 가이드 및 자동 설치
- ClawHub 스킬 검색/설치 탭 (CLI 연동)
- 포트 충돌 시 자동 대체(5177 → 5178)

## 설치

### Python 의존성

```powershell
pip install -r requirements.txt
```

### Node.js 의존성 (대시보드용)

Node.js가 설치되어 있어야 합니다. (https://nodejs.org/)

## 실행

### 메인 Launcher

```powershell
python launcher.py
```

또는

```powershell
start.bat
```

- `start.bat`는 기본적으로 콘솔 창을 숨기고(`pythonw`) 실행합니다.
- 디버그용 콘솔 모드: `set GHOSTRELAY_SHOW_CONSOLE=1` 후 `start.bat` 실행

### Skills Dashboard만 실행

```powershell
cd dashboard
node server.mjs
```

또는

```powershell
cd dashboard
start.bat
```

## 사용 방법

1. **Launcher 실행**: `python launcher.py` 또는 `start.bat`
2. **고스트 챗 열기**: 버튼 클릭 또는 트레이 메뉴
3. **창별 설정**: 고스트챗 헤더의 프로필/설정에서 모델/워크스페이스/세션키를 분리 설정
4. **게이트웨이 관리**: Launcher에서 상태 확인 후 시작/중지/재시작
5. **채팅 시작**: 입력창에 메시지를 입력하고 Enter 또는 전송 버튼 클릭
6. **Think 레벨**: 빠른 응답은 `off`, 더 깊은 추론은 `minimal~high`로 선택
7. **스킬 대시보드**: 버튼 클릭으로 웹 대시보드 열기
8. **설정**: 모델 소스를 선택하고(Vertex/Ollama) 필요한 인증/경로를 저장
9. **대화 모드**: Ollama에서 tools 미지원 모델이면 자동으로 도구 OFF 모드로 내려갑니다
10. **옵션 패널**: 고스트챗 헤더의 `옵션` 버튼에서 좌/우 패널을 토글하고 Persona/Memory/Session/Skills를 직접 수정
11. **적용 시점 안내**
   - Persona/Memory 파일 편집 저장: 즉시 파일 반영
   - Session/Memory config 토글 저장: 게이트웨이 재시작 후 안정적으로 반영

### 세션 메모리 빠른 명령

채팅창에서 아래 명령을 입력하면 모델 도구 호출 없이 로컬에서 바로 적용됩니다.

- `/session status` 또는 `/sess status`: 현재 적용 중인 세션 키 확인
- `/session reset` 또는 `/session new`: 새 런타임 세션으로 전환 (누적 컨텍스트 분리)
- `/session-memory setup` 또는 `/sm setup`: 세션 메모리 검색 설정 적용
- `/session-memory status` 또는 `/sm status`: 현재 세션 메모리 설정 조회
- `/session-memory off` 또는 `/sm off`: 세션 메모리 기능 비활성화

적용 후 게이트웨이 재시작이 필요합니다.

### 스킬 등록/관리 빠른 명령

채팅창에서 아래 명령으로 모델 도구 호출 없이 로컬에서 스킬을 관리할 수 있습니다.

- `/skill list` 또는 `/skills list`: 스킬 목록 + `skills.entries` 조회
- `/skill install <slug>`: ClawHub에서 스킬 설치 (`skills/` 폴더)
- `/skill enable <name>`: 스킬 활성화 (`skills.entries.<name>.enabled=true`)
- `/skill disable <name>`: 스킬 비활성화 (`skills.entries.<name>.enabled=false`)

설정 변경 후 게이트웨이 재시작이 필요합니다.

## ⚠️ 중요 안내

### 창/트레이 동작

- **대시보드/고스트챗 버튼 클릭** → 실행 후 자동 트레이 최소화
- **Ghost Chat 창 닫기(X)** → 콘솔 트레이로 숨김 (창은 유지)
- **콘솔 트레이 메뉴** → 창별로 열기/설정/숨기기/종료 가능
- **콘솔 트레이 메뉴 ‘종료’** → 모든 Ghost Chat 창 + 대시보드 서버 종료
  - 게이트웨이는 별도 프로세스이므로 필요 시 Launcher의 `게이트 중지` 버튼으로 종료하세요.

### 대시보드 단독 실행

- `ghostchat/dashboard/start.bat` 실행 시 대시보드만 독립 실행 가능
- 종료는 해당 콘솔을 닫으면 됩니다

## 배포

이 폴더 전체를 복사하여 배포할 수 있습니다.  
GhostRelay는 루트 `tools/clawdbot`/`utils` 없이도 동작하도록 내장 경로 탐색을 사용합니다.

### 라이선스

- 라이선스: `MIT` (`ghostchat/LICENSE`)
- 저작권자: Yun Jin Gyu (`akrnwkql@gmail.com`)
- 상업적 사용/수정/재배포 가능
- 조건: 배포본에 저작권/라이선스 고지 유지 (`LICENSE`, `NOTICE`)

### Attribution(출처 표기) 정책

- 필수: 재배포/수정 배포 시 `LICENSE`와 `NOTICE` 파일을 그대로 포함
- 권장: README 또는 배포 페이지에 아래 원문 크레딧 추가
  - `Original author: Yun Jin Gyu (akrnwkql@gmail.com)`
  - `Source: GhostRelay`

### Privacy(개인 데이터 비공개) 정책

- 공개 금지: `settings.json`, `logs/`, `workspace/`, `profiles/`, `memory/`
- 공개 금지: `.env*`, `*credentials*.json`, `*service-account*.json`, `*.pem`, `*.key`
- 공개 금지: 로컬 모델 파일(`*.gguf`, `*.safetensors`, `*.onnx`)
- 위 경로/패턴은 기본 `.gitignore`와 공개 번들 allowlist에서 제외 처리됨

### 공개 저장소용(민감정보 제외) 번들 생성

`ghostchat/release_strategy` 폴더의 allowlist 기반 스크립트를 사용하세요.

```powershell
powershell -ExecutionPolicy Bypass -File .\ghostchat\release_strategy\create_public_bundle.ps1
```

생성 결과: `ghostchat-public-bundle/`

### OpenClaw 패치형 통합 방식

`ghostchat`는 OpenClaw를 대체하지 않고 옆에 붙는 패치형 도구입니다.

```text
project-root/
  ghostchat/
  openclaw-main/
```

실행은 `ghostchat/start.bat` 기준이며, `openclaw-main`은 외부 의존성으로 유지하세요.

### Git 공개(처음 하는 경우)

1. 공개 번들 생성
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\ghostchat\release_strategy\create_public_bundle.ps1
   ```
2. 생성된 `ghostchat-public-bundle/ghostchat` 폴더만 새 공개 저장소에 업로드
3. 공개 저장소에 `LICENSE`, `NOTICE`, `.gitignore`가 포함됐는지 확인
4. 개인 데이터(`settings.json`, `logs/`, `workspace/`, `profiles/`)가 없는지 확인
5. (선택) 커밋 실수 방지:
   ```powershell
   pip install pre-commit
   pre-commit install
   ```

### 필수 요구사항

- Python 3.8+
- Node.js 18+ (대시보드용)
- OpenClaw CLI(`openclaw-main/openclaw.mjs`)가 아래 둘 중 하나에 있어야 합니다:

  1) `ghostchat` 내부에 번들
  ```
  ghostchat/
    ├── openclaw-main/
    └── ...
  ```

  2) `ghostchat`의 상위 폴더에 배치
  ```
  project-root/
    ├── ghostchat/        (이 폴더)
    └── openclaw-main/    (OpenClaw 프로젝트)
  ```

### 환경변수 (선택적)

- `OPENCLAW_WORKSPACE`: OpenClaw 워크스페이스 경로
- `OPENCLAW_CLI`: OpenClaw CLI 경로 (자동탐색 실패 시 직접 지정)
- `CLAWHUB_CLI`: ClawHub CLI 경로(필요 시 직접 지정)
- `OPENCLAW_DEFAULT_MODEL`: 게이트웨이 시작 시 기본 모델 강제 지정 (예: `ollama/llama3.3`)
- `GHOSTRELAY_OLLAMA_BASE_URL`: Ollama OpenAI 호환 엔드포인트 (예: `http://127.0.0.1:11434/v1`)
- `GHOSTRELAY_OLLAMA_API_KEY`: Ollama API 키(기본값 `ollama-local`)
- Vertex 인증은 설정창에서 저장 가능 (Service Account JSON / Project / Location)

## 주의사항

- OpenClaw 경로는 자동 탐색되며, 필요 시 `OPENCLAW_CLI`로 직접 지정할 수 있습니다
- Node.js가 설치되어 있어야 대시보드가 작동합니다
- 온보드는 최초 1회만 실행되며, 이후는 자동으로 확인합니다
- 프로필마다 `agentId`/`workspace`를 다르게 설정하면 메모리/페르소나 파일과 세션 컨텍스트를 분리할 수 있습니다
