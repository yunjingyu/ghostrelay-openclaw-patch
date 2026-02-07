"""
Ghost Chat - BRAIN 전용 채팅 인터페이스
OpenClaw Gateway와 직접 통신
"""

import sys
import os
import json
import ast
import html
import subprocess
import time
import uuid
import websocket
import re
import shlex
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QTextBrowser, QLineEdit, QPushButton, QLabel, QSystemTrayIcon, QMenu, QSlider, QStyle,
    QCheckBox, QSizeGrip, QComboBox, QFrame, QTabWidget, QTableWidget, QTableWidgetItem, QDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QPoint, QTimer, QRect, QEvent, QFileSystemWatcher
from PySide6.QtGui import QTextCursor, QAction


sys.path.insert(0, str(Path(__file__).parent))

from local_logger import get_logger
from runtime_paths import resolve_gateway_script, resolve_openclaw_dir, resolve_project_root
from settings_store import (
    get_active_profile_id,
    get_profile,
    list_profiles,
    load_settings,
    normalize_provider_id,
    normalize_profile_id,
    save_settings,
)

logger = get_logger("ghost_chat")

LEGACY_SESSION_KEYS = {"main", "default"}
_RUNTIME_SESSION_KEY = f"ghostrelay-{int(time.time())}-{os.getpid()}"
AGENT_ID_SANITIZE_RE = re.compile(r"[^a-z0-9_-]+")
PROFILE_BOOTSTRAP_FILES = (
    "AGENTS.md",
    "SOUL.md",
    "IDENTITY.md",
    "USER.md",
    "TOOLS.md",
    "HEARTBEAT.md",
    "MEMORY.md",
)
DEFAULT_OLLAMA_HTTP_TIMEOUT_SEC = 180
OLLAMA_DEFAULT_HISTORY_TURNS = 6
INJECTION_PRESET_LIMITS = {
    "weak": (900, 600, 800),
    "medium": (1800, 1200, 1600),
    "strong": (3200, 2200, 2800),
}
_OLLAMA_SHOW_CACHE: dict[str, dict[str, Any]] = {}


def generate_runtime_session_key() -> str:
    return f"ghostrelay-{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


def normalize_agent_id(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    raw = AGENT_ID_SANITIZE_RE.sub("-", raw).strip("-")
    return raw or "main"


def resolve_workspace_path(repo_root: Path, value: str | None) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return (repo_root / "workspace").resolve()
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / raw).resolve()


def build_agent_session_key(agent_id: str, raw_session_key: str) -> str:
    normalized_agent = normalize_agent_id(agent_id)
    key = str(raw_session_key or "").strip().lower()
    if not key:
        key = generate_runtime_session_key()
    if key.startswith("agent:"):
        return key
    return f"agent:{normalized_agent}:{key}"


def decode_process_output(data: bytes | None) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", "cp949", "cp1252"):
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def run_process_capture(
    args: list[str],
    cwd: str | None = None,
    timeout: int | None = None,
    env: dict | None = None,
) -> tuple[int, str, str]:
    subprocess_kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if creation_flags:
            subprocess_kwargs["creationflags"] = creation_flags
        if hasattr(subprocess, "STARTUPINFO"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
            subprocess_kwargs["startupinfo"] = startupinfo

    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=False,
        timeout=timeout,
        env=env,
        **subprocess_kwargs,
    )
    stdout_text = decode_process_output(result.stdout)
    stderr_text = decode_process_output(result.stderr)
    return result.returncode, stdout_text, stderr_text


def hidden_popen_kwargs() -> dict[str, Any]:
    if sys.platform != "win32":
        return {}
    kwargs: dict[str, Any] = {}
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if creation_flags:
        kwargs["creationflags"] = creation_flags
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        kwargs["startupinfo"] = startupinfo
    return kwargs


def resolve_ollama_cli_path() -> Optional[str]:
    env_cli = str(os.environ.get("OLLAMA_CLI") or "").strip()
    if env_cli and Path(env_cli).exists():
        return str(Path(env_cli))
    detected = shutil.which("ollama")
    if detected:
        return str(Path(detected))
    if sys.platform == "win32":
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe",
        ]
        for candidate in candidates:
            if str(candidate).strip() and candidate.exists():
                return str(candidate)
    return None


class OpenClawGatewayManager:
    """OpenClaw Gateway 및 온보드 관리"""
    
    def __init__(self):
        self.ghostchat_dir = Path(__file__).resolve().parent
        self.openclaw_dir = resolve_openclaw_dir()
        self.repo_root = resolve_project_root()
        self.gateway_bat = resolve_gateway_script()
        self.gateway_process: Optional[subprocess.Popen] = None
        self.gateway_token: Optional[str] = None
        self.settings = load_settings()

    def get_active_agent_id(self) -> str:
        settings = self.settings if isinstance(self.settings, dict) else {}
        return normalize_agent_id(str(settings.get("agentId") or "main"))

    def get_active_workspace(self) -> Path:
        settings = self.settings if isinstance(self.settings, dict) else {}
        return resolve_workspace_path(self.repo_root, settings.get("workspace"))

    def read_agents_list(self, openclaw_mjs: Path) -> list[dict]:
        try:
            code, stdout_text, _ = run_process_capture(
                ["node", str(openclaw_mjs), "config", "get", "--json", "agents.list"],
                cwd=str(self.openclaw_dir),
                timeout=20,
            )
            if code != 0:
                return []
            parsed = json.loads(stdout_text or "[]")
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            return []
        return []

    def config_set(self, openclaw_mjs: Path, path: str, value: str, as_json: bool = False) -> tuple[bool, str]:
        args = ["node", str(openclaw_mjs), "config", "set"]
        if as_json:
            args.append("--json")
        args.extend([path, value])
        code, stdout_text, stderr_text = run_process_capture(
            args,
            cwd=str(self.openclaw_dir),
            timeout=25,
        )
        output = stdout_text.strip() or stderr_text.strip()
        return code == 0, output

    def ensure_active_agent_config(
        self,
        issues: Optional[list[str]] = None,
        outputs: Optional[list[str]] = None,
    ) -> bool:
        openclaw_mjs = self.openclaw_dir / "openclaw.mjs"
        if not openclaw_mjs.exists():
            if issues is not None:
                issues.append(f"openclaw.mjs not found: {openclaw_mjs}")
            return False

        settings = self.settings if isinstance(self.settings, dict) else {}
        agent_id = normalize_agent_id(str(settings.get("agentId") or "main"))
        model = str(settings.get("model") or "").strip() or "google-vertex/gemini-2.0-flash"
        workspace_dir = self.get_active_workspace()
        workspace_dir.mkdir(parents=True, exist_ok=True)

        agent_entries = self.read_agents_list(openclaw_mjs)
        index = -1
        for idx, entry in enumerate(agent_entries):
            if normalize_agent_id(str(entry.get("id") or "")) == agent_id:
                index = idx
                break
        if index < 0:
            index = len(agent_entries)

        updates = [
            (f"agents.list[{index}].id", agent_id, False),
            (f"agents.list[{index}].workspace", str(workspace_dir), False),
            (f"agents.list[{index}].model.primary", model, False),
        ]
        all_ok = True
        for path, value, as_json in updates:
            ok, output = self.config_set(openclaw_mjs, path, value, as_json=as_json)
            all_ok = all_ok and ok
            if outputs is not None and output:
                outputs.append(output)
            if not ok:
                logger.warning(f"agent config set failed: {path} => {output}")
                if issues is not None:
                    issues.append(f"{path}: {output}")
        return all_ok

    def resolve_state_dir(self) -> Path:
        raw = str(os.environ.get("OPENCLAW_STATE_DIR") or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return (Path.home() / ".openclaw").resolve()

    def resolve_agent_auth_profile_path(self, agent_id: str) -> Path:
        normalized = normalize_agent_id(agent_id)
        return self.resolve_state_dir() / "agents" / normalized / "agent" / "auth-profiles.json"

    def ensure_workspace_scaffold(self, workspace_dir: Path, profile_label: str) -> list[str]:
        details: list[str] = []
        normalized_profile = normalize_agent_id(profile_label)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "memory").mkdir(parents=True, exist_ok=True)
        main_workspace = (self.repo_root / "workspace").resolve()

        for name in PROFILE_BOOTSTRAP_FILES:
            target = workspace_dir / name
            if target.exists():
                continue
            copied = False
            source = main_workspace / name
            if normalized_profile != "main" and source.exists():
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    copied = True
                    details.append(f"seeded {name} from main workspace")
                except Exception as e:
                    details.append(f"failed to seed {name}: {e}")
            if copied:
                continue
            try:
                target.write_text(
                    f"# {name} ({normalized_profile})\n\n- profile bootstrap file\n",
                    encoding="utf-8",
                )
                details.append(f"created {name}")
            except Exception as e:
                details.append(f"failed to create {name}: {e}")
        return details

    def ensure_agent_auth_profile(self, agent_id: str) -> tuple[bool, str]:
        target = self.resolve_agent_auth_profile_path(agent_id)
        source = self.resolve_agent_auth_profile_path("main")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size > 0:
                return True, "auth-profiles already present"
            if source.exists() and source.stat().st_size > 0:
                shutil.copy2(source, target)
                return True, "auth-profiles synced from main agent"
            target.write_text('{"version":1,"profiles":{}}', encoding="utf-8")
            return True, "auth-profiles initialized (empty)"
        except Exception as e:
            return False, str(e)

    def ensure_ollama_auth_profile(self, agent_id: str) -> tuple[bool, str]:
        settings = self.settings if isinstance(self.settings, dict) else {}
        model_source = str(settings.get("modelSource") or "").strip().lower()
        if model_source != "ollama":
            return True, "ollama auth not required"
        ollama = settings.get("ollama", {}) if isinstance(settings.get("ollama"), dict) else {}
        api_key = str(ollama.get("apiKey") or "").strip() or "ollama-local"
        target = self.resolve_agent_auth_profile_path(agent_id)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raw = target.read_text(encoding="utf-8")
                store = json.loads(raw or "{}")
            else:
                store = {}
            if not isinstance(store, dict):
                store = {}
            store["version"] = int(store.get("version") or 1)
            profiles = store.get("profiles")
            if not isinstance(profiles, dict):
                profiles = {}
            profiles["ollama:ghostrelay"] = {
                "type": "api_key",
                "provider": "ollama",
                "key": api_key,
            }
            store["profiles"] = profiles
            target.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
            return True, "ollama auth profile ensured"
        except Exception as e:
            return False, str(e)

    def sync_active_profile_runtime(self) -> tuple[bool, list[str], bool]:
        details: list[str] = []
        issues: list[str] = []
        outputs: list[str] = []
        settings = self.settings if isinstance(self.settings, dict) else {}
        agent_id = normalize_agent_id(str(settings.get("agentId") or "main"))
        workspace_dir = self.get_active_workspace()
        details.extend(self.ensure_workspace_scaffold(workspace_dir, agent_id))
        self.ensure_active_agent_config(issues=issues, outputs=outputs)
        auth_ok, auth_msg = self.ensure_agent_auth_profile(agent_id)
        details.append(auth_msg)
        if not auth_ok:
            issues.append(f"auth profile sync failed: {auth_msg}")
        ollama_ok, ollama_msg = self.ensure_ollama_auth_profile(agent_id)
        details.append(ollama_msg)
        if not ollama_ok:
            issues.append(f"ollama auth profile sync failed: {ollama_msg}")
        restart_hint = any("Restart the gateway" in line for line in outputs)
        return len(issues) == 0, details + issues, restart_hint

    def apply_vertex_env(self, env: dict) -> dict:
        settings = self.settings if isinstance(self.settings, dict) else {}
        model_source = str(settings.get("modelSource") or "vertex").strip().lower()
        model_ref = str(settings.get("model") or "").strip()
        if model_ref:
            env["OPENCLAW_DEFAULT_MODEL"] = model_ref
        workspace_dir = self.get_active_workspace()
        env["OPENCLAW_WORKSPACE"] = str(workspace_dir)

        if model_source == "ollama":
            ollama = settings.get("ollama", {}) if isinstance(settings.get("ollama"), dict) else {}
            api_key = str(ollama.get("apiKey") or "").strip() or "ollama-local"
            base_url = str(ollama.get("baseUrl") or "").strip()
            env["OLLAMA_API_KEY"] = api_key
            env["GHOSTRELAY_OLLAMA_API_KEY"] = api_key
            if base_url:
                env["GHOSTRELAY_OLLAMA_BASE_URL"] = base_url
            return env

        if model_source == "api":
            api_provider = (
                settings.get("apiProvider", {})
                if isinstance(settings.get("apiProvider"), dict)
                else {}
            )
            provider_id = normalize_provider_id(str(api_provider.get("providerId") or ""), "openai")
            base_url = str(api_provider.get("baseUrl") or "").strip()
            api_key = str(api_provider.get("apiKey") or "").strip()
            env["GHOSTRELAY_API_PROVIDER_ID"] = provider_id
            if base_url:
                env["GHOSTRELAY_API_BASE_URL"] = base_url
            if api_key:
                env["GHOSTRELAY_API_KEY"] = api_key
            return env

        vertex = settings.get("vertex", {}) if isinstance(settings.get("vertex"), dict) else {}
        sa_path = str(vertex.get("serviceAccountPath") or "").strip()
        project = str(vertex.get("project") or "").strip()
        location = str(vertex.get("location") or "").strip()
        if sa_path:
            env["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        if project:
            env["GOOGLE_CLOUD_PROJECT"] = project
            env["GCLOUD_PROJECT"] = project
        if location:
            env["GOOGLE_CLOUD_LOCATION"] = location
        return env
        
    def check_gateway_running(self) -> bool:
        """게이트웨이 실행 중인지 확인"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock_v4:
                # UI 폴링 지연을 줄이기 위해 localhost IPv4만 빠르게 확인한다.
                sock_v4.settimeout(0.05)
                return sock_v4.connect_ex(("127.0.0.1", 18789)) == 0
        except Exception:
            return False
    
    def start_gateway(self) -> bool:
        """게이트웨이 시작"""
        if not isinstance(self.settings, dict) or not self.settings:
            self.settings = load_settings()
        self.ensure_active_agent_config()
        if self.check_gateway_running():
            logger.info("게이트웨이가 이미 실행 중입니다")
            return True
            
        try:
            openclaw_mjs = self.openclaw_dir / "openclaw.mjs"
            if not openclaw_mjs.exists():
                logger.error(f"OpenClaw CLI를 찾을 수 없습니다: {openclaw_mjs}")
                return False

            # Windows 배치 스크립트 우선 실행 (빌드/설정 포함)
            if sys.platform == "win32" and self.gateway_bat.exists():
                logger.info(f"gateway script 실행 시도: {self.gateway_bat}")
                try:
                    env = os.environ.copy()
                    env = self.apply_vertex_env(env)
                    subprocess.Popen(
                        ["cmd", "/c", str(self.gateway_bat)],
                        cwd=str(self.gateway_bat.parent),
                        env=env,
                        **hidden_popen_kwargs(),
                    )
                    # 포트 연결 대기 (최대 30초)
                    for i in range(60):
                        if self.check_gateway_running():
                            logger.info("게이트웨이 시작 완료 (bat)")
                            return True
                        if i % 10 == 0 and i > 0:
                            logger.info(f"게이트웨이 시작 대기 중... ({i * 0.5:.1f}초 경과)")
                        time.sleep(0.5)
                    logger.warning("게이트웨이 시작 타임아웃 (bat)")
                except Exception as e:
                    logger.warning(f"start_gateway.bat 실행 실패: {e}")
            
            # [FIX] Gateway start blocked 방지: gateway.mode local 강제
            try:
                logger.info("gateway.mode를 local로 설정 중...")
                run_process_capture(
                    ["node", str(openclaw_mjs), "config", "set", "gateway.mode", "local"],
                    cwd=str(self.openclaw_dir),
                    timeout=20,
                )
            except Exception as e:
                logger.warning(f"gateway.mode 설정 실패 (무시 가능): {e}")

            # gateway.auth.token 설정 (token 모드 요구 시 필수)
            try:
                token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
                if not token:
                    import uuid
                    token = str(uuid.uuid4())
                    os.environ["OPENCLAW_GATEWAY_TOKEN"] = token
                self.gateway_token = token
                run_process_capture(
                    ["node", str(openclaw_mjs), "config", "set", "gateway.auth.mode", "token"],
                    cwd=str(self.openclaw_dir),
                    timeout=20,
                )
                run_process_capture(
                    ["node", str(openclaw_mjs), "config", "set", "gateway.auth.token", token],
                    cwd=str(self.openclaw_dir),
                    timeout=20,
                )
            except Exception as e:
                logger.warning(f"gateway.auth.token 설정 실패 (무시 가능): {e}")
            
            # 게이트웨이 시작 (백그라운드)
            logger.info("게이트웨이 시작 중...")
            # gateway run 명령 사용 (gateway start는 서비스 설치용)
            gateway_cmd = [
                "node", str(openclaw_mjs),
                "gateway", "run",
                "--dev",
                "--allow-unconfigured",
                "--force",
                "--port", "18789",
                "--bind", "loopback"
            ]
            
            logger.info(f"게이트웨이 명령: {' '.join(gateway_cmd)}")
            
            self.gateway_process = subprocess.Popen(
                gateway_cmd,
                cwd=str(self.openclaw_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **hidden_popen_kwargs(),
            )
            
            # 프로세스 시작 확인 (1초 대기)
            time.sleep(1)
            if self.gateway_process.poll() is not None:
                # 프로세스가 즉시 종료됨 = 오류 발생
                try:
                    stdout, stderr = self.gateway_process.communicate(timeout=2)
                    error_msg = ""
                    if stderr:
                        error_msg += f"STDERR: {stderr.decode('utf-8', errors='ignore')}\n"
                    if stdout:
                        error_msg += f"STDOUT: {stdout.decode('utf-8', errors='ignore')}\n"
                    if not error_msg:
                        error_msg = f"프로세스가 종료 코드 {self.gateway_process.returncode}로 즉시 종료됨"
                    logger.error(f"게이트웨이 시작 실패:\n{error_msg}")
                except:
                    logger.error(f"게이트웨이 프로세스가 즉시 종료됨 (종료 코드: {self.gateway_process.returncode})")
                return False
            
            # 게이트웨이 시작 대기 (최대 30초, 더 긴 대기)
            logger.info("게이트웨이 시작 대기 중... (최대 30초)")
            for i in range(60):  # 60번 * 0.5초 = 30초
                if self.check_gateway_running():
                    logger.info("게이트웨이 시작 완료")
                    return True
                if i % 10 == 0 and i > 0:  # 5초마다 로그
                    logger.info(f"게이트웨이 시작 대기 중... ({i * 0.5:.1f}초 경과)")
                time.sleep(0.5)
            
            logger.warning("게이트웨이 시작 타임아웃 (30초)")
            # 프로세스가 실행 중인지 확인
            if self.gateway_process and self.gateway_process.poll() is None:
                logger.info("게이트웨이 프로세스는 실행 중이지만 포트 연결 실패. 잠시 후 다시 확인하세요.")
            return False
            
        except Exception as e:
            logger.error(f"게이트웨이 시작 실패: {e}")
            return False

    def get_gateway_token(self) -> Optional[str]:
        """설정된 게이트웨이 토큰 조회"""
        if self.gateway_token:
            return self.gateway_token
        token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
        if token:
            self.gateway_token = token
            return token
        # Fast path: read config file first (avoids node process startup latency on send).
        try:
            config_path = Path.home() / ".openclaw" / "openclaw.json"
            if config_path.exists():
                data = json.loads(config_path.read_text(encoding="utf-8"))
                token = (
                    data.get("gateway", {})
                        .get("auth", {})
                        .get("token")
                )
                if token:
                    self.gateway_token = str(token)
                    os.environ["OPENCLAW_GATEWAY_TOKEN"] = self.gateway_token
                    return self.gateway_token
        except Exception:
            pass
        try:
            openclaw_mjs = self.openclaw_dir / "openclaw.mjs"
            return_code, stdout_text, _ = run_process_capture(
                ["node", str(openclaw_mjs), "config", "get", "gateway.auth.token"],
                cwd=str(self.openclaw_dir),
                timeout=3,
            )
            if return_code == 0:
                token = stdout_text.strip()
                if token:
                    self.gateway_token = token
                    os.environ["OPENCLAW_GATEWAY_TOKEN"] = self.gateway_token
                    return token
        except Exception:
            pass
        return None
    
    def check_onboard(self) -> bool:
        """온보드 완료 여부 확인"""
        try:
            openclaw_mjs = self.openclaw_dir / "openclaw.mjs"
            config_path = Path.home() / ".openclaw" / "openclaw.json"
            if config_path.exists():
                return True
            return_code, _, _ = run_process_capture(
                ["node", str(openclaw_mjs), "config", "get", "gateway.mode"],
                cwd=str(self.openclaw_dir),
                timeout=5
            )
            return return_code == 0
        except Exception:
            return False
    
    def run_onboard(self) -> bool:
        """온보드 실행"""
        if self.check_onboard():
            logger.info("온보드가 이미 완료되었습니다")
            return True
            
        try:
            openclaw_mjs = self.openclaw_dir / "openclaw.mjs"
            logger.info("온보드 실행 중 (interactive)...")
            if sys.platform == "win32":
                try:
                    subprocess.Popen(
                        ["cmd", "/c", "start", "OpenClaw Onboard", "node", str(openclaw_mjs), "onboard"],
                        cwd=str(self.openclaw_dir),
                    )
                    logger.info("온보드(인터랙티브) 창을 열었습니다. Yes 입력 후 완료하세요.")
                except Exception as e:
                    logger.error(f"온보드 창 열기 실패: {e}")
                    return False
            else:
                subprocess.Popen(
                    ["node", str(openclaw_mjs), "onboard"],
                    cwd=str(self.openclaw_dir),
                )
            return False
        except Exception as e:
            logger.error(f"온보드 실행 실패: {e}")
            return False


class GatewayStartupThread(QThread):
    """게이트웨이/온보드 시작 스레드 (UI 스레드 분리)"""
    log_line = Signal(str)
    status_changed = Signal(str)
    startup_ready = Signal()
    startup_failed = Signal(str)

    def __init__(self, gateway_manager: OpenClawGatewayManager):
        super().__init__()
        self.gateway_manager = gateway_manager

    def run(self):
        try:
            # Fast path: if gateway is already running, attach immediately.
            if self.gateway_manager.check_gateway_running():
                self.log_line.emit("✅ 기존 게이트웨이 감지: 즉시 연결합니다.")
                self.status_changed.emit("✅ 준비 완료")
                self.startup_ready.emit()
                return

            self.status_changed.emit("게이트웨이 시작 중...")
            self.log_line.emit("🚀 Ghost Chat 시작 중...")
            self.log_line.emit("⏳ 게이트웨이 시작 중... (최대 30초 소요)")

            gateway_started = self.gateway_manager.start_gateway()
            if not gateway_started:
                self.startup_failed.emit("게이트웨이 시작 실패")
                return

            self.log_line.emit("✅ 게이트웨이 시작 완료")
            self.status_changed.emit("온보드 확인 중...")

            if not self.gateway_manager.check_onboard():
                self.log_line.emit("📋 온보드 창을 열었습니다. 콘솔에서 Yes를 입력하세요.")
                self.gateway_manager.run_onboard()
                for i in range(240):
                    if self.gateway_manager.check_onboard():
                        self.log_line.emit("✅ 온보드 완료")
                        break
                    if i % 20 == 0 and i > 0:
                        self.log_line.emit(f"⏳ 온보드 대기 중... ({i * 0.5:.1f}초)")
                    self.msleep(500)
                else:
                    self.status_changed.emit("⚠️ 온보드 미완료")
                    self.log_line.emit("⚠️ 온보드가 완료되지 않았습니다. 콘솔에서 온보드를 끝내주세요.")

            self.status_changed.emit("✅ 준비 완료")
            self.log_line.emit("✅ 준비 완료! 채팅을 시작하세요.")
            self.startup_ready.emit()
        except Exception as e:
            self.startup_failed.emit(str(e))


class ProfileRuntimeSyncThread(QThread):
    completed = Signal(bool, object, bool)

    def __init__(self, profile_settings: dict):
        super().__init__()
        self.profile_settings = profile_settings if isinstance(profile_settings, dict) else {}

    def run(self):
        manager = OpenClawGatewayManager()
        manager.settings = dict(self.profile_settings)
        ok, details, restart_hint = manager.sync_active_profile_runtime()
        self.completed.emit(ok, details, restart_hint)


class OpenClawAgentThread(QThread):
    """OpenClaw Agent 실행 스레드"""
    message_received = Signal(str)  # message
    debug_received = Signal(str)
    partial_received = Signal(str)
    stats_received = Signal(object)
    
    def __init__(
        self,
        user_input: str,
        gateway_token: Optional[str],
        history: list[dict],
        session_key: Optional[str] = None,
        agent_id: str = "main",
        profile_settings: Optional[dict] = None,
        thinking_mode: str = "off",
    ):
        super().__init__()
        self.user_input = user_input
        self.gateway_token = gateway_token
        self.history = history
        self.session_key = (session_key or "").strip()
        self.agent_id = normalize_agent_id(agent_id)
        self.profile_settings = profile_settings if isinstance(profile_settings, dict) else {}
        raw_tool_mode = str(self.profile_settings.get("toolMode") or "auto").strip().lower()
        self.tool_mode = raw_tool_mode if raw_tool_mode in {"auto", "chat", "agent"} else "auto"
        self.thinking_mode = thinking_mode if thinking_mode in {
            "off", "minimal", "low", "medium", "high"
        } else "off"
        self.ghostchat_dir = Path(__file__).resolve().parent
        self.openclaw_dir = resolve_openclaw_dir()
        self.repo_root = resolve_project_root()

    def resolve_session_key(self) -> str:
        if self.session_key:
            return build_agent_session_key(self.agent_id, self.session_key)
        try:
            settings = load_settings()
            if isinstance(settings, dict):
                key = str(settings.get("sessionKey") or "").strip()
                if key and key.lower() not in LEGACY_SESSION_KEYS:
                    return build_agent_session_key(self.agent_id, key)
        except Exception:
            pass
        return build_agent_session_key(self.agent_id, _RUNTIME_SESSION_KEY)

    def get_profile_model_source(self) -> str:
        source = str(self.profile_settings.get("modelSource") or "").strip().lower()
        return source if source in {"vertex", "ollama", "api"} else "vertex"

    def get_profile_ollama_tools_support(self) -> Optional[bool]:
        if self.get_profile_model_source() != "ollama":
            return None
        ollama = self.profile_settings.get("ollama", {})
        if not isinstance(ollama, dict):
            return None
        value = ollama.get("supportsTools")
        return value if isinstance(value, bool) else None

    def resolve_dispatch_mode(self) -> str:
        # chat: tool-call 없이 HTTP(OpenAI-compatible) 경로
        # agent: WS agent 경로(도구 호출 가능)
        if self.tool_mode == "chat":
            return "chat"
        if self.tool_mode == "agent":
            return "agent"
        if self.get_profile_model_source() == "ollama":
            support = self.get_profile_ollama_tools_support()
            if support is True:
                return "agent"
            return "chat"
        return "agent"

    def is_heartbeat_like(self, text: Optional[str]) -> bool:
        if not text:
            return True
        normalized = text.strip().upper()
        return normalized == "HEARTBEAT_OK" or normalized.startswith("HEARTBEAT_OK ")

    def build_retry_message(self) -> str:
        raw = (self.user_input or "").replace("\n", " ").strip()
        if not raw:
            raw = "(empty)"
        return (
            f'사용자 입력: "{raw}"\n'
            "위 입력이 짧거나 모호하더라도 HEARTBEAT_OK를 출력하지 말고 "
            "한국어로 한 문장 이상 답하거나 필요한 확인 질문을 한 개 해주세요."
        )

    def clean_response_text(self, text: Optional[str]) -> str:
        raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            return ""
        lines: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if not stripped:
                continue
            if lowered in {"<think>", "</think>", "<thinking>", "</thinking>", "<analysis>", "</analysis>"}:
                continue
            if lowered in {"heartbeat", "heartbeat:", "_heartbeat ok!_", "heartbeat_ok", "heartbeat ok"}:
                continue
            if stripped in {"지금 번역하기", "번역결과"}:
                continue
            if lowered.startswith("#context_"):
                continue
            lines.append(stripped)
        cleaned = "\n".join(lines).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def is_contaminated_response(self, text: Optional[str]) -> bool:
        raw = (text or "").strip()
        if not raw:
            return True
        lowered = raw.lower()
        if lowered.count("heartbeat") >= 3:
            return True
        if "지금 번역하기" in raw and "번역결과" in raw:
            return True
        if "사용자 입력:" in raw and "heartbeat" in lowered:
            return True
        if "</think>" in lowered and "<think>" not in lowered:
            return True
        return False

    def repair_chat_response(
        self,
        response_text: Optional[str],
        token: str,
        session_key: str,
        use_direct_ollama: bool,
    ) -> str:
        cleaned = self.clean_response_text(response_text)
        if cleaned and not self.is_contaminated_response(cleaned):
            return cleaned

        self.debug_received.emit("응답 오염 감지. 정제 재요청을 시도합니다.")
        repair_prompt = (
            "You are in normal end-user chat mode. "
            "Answer only the user's original message in Korean. "
            "Do not output HEARTBEAT, translations, tool-call text, or think tags."
        )
        repair_message = (
            f'원래 사용자 질문: "{self.user_input}"\n'
            "내부 메모/포맷/번역 문구는 모두 무시하고 사용자에게 보여줄 최종 답변만 "
            "한국어 2~4문장으로 작성하세요."
        )
        try:
            if use_direct_ollama:
                refined, refined_stats = self.send_via_direct_ollama_http(repair_message, repair_prompt)
                if isinstance(refined_stats, dict):
                    self.stats_received.emit(refined_stats)
            else:
                refined = self.send_via_openai_http(token, session_key, repair_message, repair_prompt)
        except Exception as e:
            self.debug_received.emit(f"정제 재요청 실패: {e}")
            return cleaned or (response_text or "").strip()

        if self.is_heartbeat_like(refined):
            return cleaned or (response_text or "").strip()
        refined_clean = self.clean_response_text(refined)
        if refined_clean:
            return refined_clean
        return cleaned or (response_text or "").strip()

    def should_preemptive_disambiguate(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return True
        # Very short or symbol-only inputs often trigger "no-action" logic in workspace rules.
        if len(raw) <= 3:
            return True
        if re.fullmatch(r"[\W_]+", raw, flags=re.UNICODE):
            return True
        return False

    def is_tool_call_like(self, text: Optional[str]) -> bool:
        if not text:
            return False
        raw = text.strip().strip("`").strip()
        if raw.startswith("```") and raw.endswith("```"):
            raw = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw).strip()
        if not raw:
            return False
        if "\n" in raw:
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            if len(lines) != 1:
                return False
            raw = lines[0]
        return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*;?", raw) is not None

    def extract_text_from_payload_blob(self, blob: str) -> Optional[str]:
        payload_obj = None
        raw = (blob or "").strip()
        if not raw or not raw.startswith("{"):
            return None
        try:
            payload_obj = json.loads(raw)
        except Exception:
            try:
                payload_obj = ast.literal_eval(raw)
            except Exception:
                return None
        if not isinstance(payload_obj, dict):
            return None
        result = payload_obj.get("result")
        if not isinstance(result, dict):
            return None
        payloads = result.get("payloads")
        if not isinstance(payloads, list):
            return None
        texts: list[str] = []
        for item in payloads:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item.get("text")))
        merged = "\n".join(part.strip() for part in texts if str(part).strip()).strip()
        return merged or None

    def is_provider_rate_limit_error(self, text: Optional[str]) -> bool:
        if not text:
            return False
        upper = text.upper()
        return (
            "RESOURCE_EXHAUSTED" in upper
            or "LLM ERROR" in upper and "429" in upper
            or "\"CODE\": 429" in upper
            or "CODE\": 429" in upper
        )

    def is_provider_error_text(self, text: Optional[str]) -> bool:
        if not text:
            return False
        upper = text.upper()
        return "LLM ERROR" in upper or "RESOURCE_EXHAUSTED" in upper

    def is_tools_unsupported_error(self, text: Optional[str]) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return (
            "does not support tools" in lowered
            or "tools are not supported" in lowered
            or ("tool calling" in lowered and "not support" in lowered)
            or ("tool calls" in lowered and "not support" in lowered)
        )

    def get_active_provider_id(self) -> str:
        source_settings = self.profile_settings if isinstance(self.profile_settings, dict) else {}
        model_ref = str(source_settings.get("model") or "").strip().lower()
        if "/" in model_ref:
            return normalize_provider_id(model_ref.split("/", 1)[0], "openai")
        source = str(source_settings.get("modelSource") or "").strip().lower()
        if source == "ollama":
            return "ollama"
        if source == "api":
            api_provider = (
                source_settings.get("apiProvider", {})
                if isinstance(source_settings.get("apiProvider"), dict)
                else {}
            )
            return normalize_provider_id(str(api_provider.get("providerId") or ""), "openai")
        if source == "vertex":
            return "google-vertex"
        return ""

    def read_agents_list(self, openclaw_mjs: Path) -> list[dict]:
        try:
            code, stdout_text, _ = run_process_capture(
                ["node", str(openclaw_mjs), "config", "get", "--json", "agents.list"],
                cwd=str(openclaw_mjs.parent),
                timeout=20,
            )
            if code != 0:
                return []
            parsed = json.loads(stdout_text or "[]")
            if isinstance(parsed, list):
                return [entry for entry in parsed if isinstance(entry, dict)]
        except Exception:
            return []
        return []

    def apply_provider_no_tools_policy(self) -> tuple[bool, str]:
        openclaw_mjs = self.openclaw_dir / "openclaw.mjs"
        if not openclaw_mjs.exists():
            return False, f"openclaw.mjs not found: {openclaw_mjs}"
        provider_id = self.get_active_provider_id()
        if not provider_id:
            return False, "provider id could not be resolved"

        agents = self.read_agents_list(openclaw_mjs)
        agent_index = -1
        for idx, entry in enumerate(agents):
            if normalize_agent_id(str(entry.get("id") or "")) == self.agent_id:
                agent_index = idx
                break
        if agent_index < 0:
            return False, f"agent not found in agents.list: {self.agent_id}"

        path = f"agents.list[{agent_index}].tools.byProvider.{provider_id}.allow"
        code, stdout_text, stderr_text = run_process_capture(
            ["node", str(openclaw_mjs), "config", "set", "--json", path, "[]"],
            cwd=str(openclaw_mjs.parent),
            timeout=25,
        )
        output = (stdout_text.strip() or stderr_text.strip() or "").strip()
        if code != 0:
            return False, output or f"failed to set {path}"
        return True, output or f"applied {path}=[]"

    def build_openai_messages(
        self,
        user_message: str,
        extra_system_prompt: Optional[str] = None,
    ) -> list[dict]:
        messages: list[dict] = []
        system_prompt = "This is an interactive user chat. Always answer the user directly. Do NOT reply with HEARTBEAT_OK."
        if extra_system_prompt:
            system_prompt = f"{system_prompt}\n{extra_system_prompt}"
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )
        messages.append({"role": "user", "content": user_message})
        return messages

    def get_ollama_settings(self) -> dict:
        settings = self.profile_settings if isinstance(self.profile_settings, dict) else {}
        ollama = settings.get("ollama", {})
        return ollama if isinstance(ollama, dict) else {}

    def get_ollama_injection_level(self) -> str:
        ollama = self.get_ollama_settings()
        level = str(ollama.get("injectionLevel") or "medium").strip().lower()
        return level if level in {"weak", "medium", "strong"} else "medium"

    def get_ollama_history_turns(self) -> int:
        ollama = self.get_ollama_settings()
        return clamp_int(
            ollama.get("historyTurns"),
            OLLAMA_DEFAULT_HISTORY_TURNS,
            1,
            20,
        )

    def get_injection_limits(self) -> tuple[int, int, int]:
        level = self.get_ollama_injection_level()
        limits = INJECTION_PRESET_LIMITS.get(level)
        if not limits:
            limits = INJECTION_PRESET_LIMITS["medium"]
        return limits

    def get_profile_workspace_dir(self) -> Path:
        profile = self.profile_settings if isinstance(self.profile_settings, dict) else {}
        return resolve_workspace_path(self.repo_root, profile.get("workspace"))

    def read_workspace_text(self, relative_path: str, max_chars: int) -> str:
        target = self.get_profile_workspace_dir() / relative_path
        try:
            if not target.exists():
                return ""
            text = target.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                return ""
            if len(text) > max_chars:
                return f"{text[:max_chars]}\n...(truncated)"
            return text
        except Exception:
            return ""

    def is_first_turn_in_session(self) -> bool:
        history_items = [
            item for item in self.history
            if isinstance(item, dict)
            and str(item.get("role") or "").strip() in {"user", "assistant"}
            and str(item.get("content") or "").strip()
        ]
        return len(history_items) <= 1

    def tokenize_query(self, text: str) -> list[str]:
        raw = str(text or "").lower()
        tokens = re.findall(r"[a-z0-9가-힣_]{2,}", raw)
        deduped: list[str] = []
        for token in tokens:
            if token not in deduped:
                deduped.append(token)
        return deduped[:18]

    def estimate_token_count(self, text: str) -> int:
        raw = str(text or "")
        if not raw.strip():
            return 0
        cjk_count = len(re.findall(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af\u4e00-\u9fff\u3040-\u30ff]", raw))
        latin_count = len(re.findall(r"[A-Za-z0-9]", raw))
        other_count = max(0, len(raw) - cjk_count - latin_count)
        estimated = int(round((cjk_count * 1.05) + (latin_count / 4.0) + (other_count / 2.6)))
        return max(1, estimated)

    def _pick_int_by_key_fragments(self, model_info: dict[str, Any], fragments: list[str]) -> Optional[int]:
        for key, value in model_info.items():
            lower_key = str(key).lower()
            if not all(fragment in lower_key for fragment in fragments):
                continue
            try:
                parsed = int(float(str(value).strip()))
            except Exception:
                continue
            if parsed > 0:
                return parsed
        return None

    def get_ollama_model_info_cached(self, base_url: str, model_name: str) -> dict[str, Any]:
        import urllib.request

        normalized_base = (base_url or "").strip().rstrip("/")
        if not normalized_base:
            normalized_base = "http://127.0.0.1:11434/v1"
        if normalized_base.endswith("/v1"):
            normalized_base = normalized_base[:-3]
        normalized_model = str(model_name or "").strip().lower()
        cache_key = f"{normalized_base}|{normalized_model}"
        cached = _OLLAMA_SHOW_CACHE.get(cache_key)
        if isinstance(cached, dict):
            return cached

        show_url = f"{normalized_base}/api/show"
        payload = json.dumps({"model": model_name}).encode("utf-8")
        req = urllib.request.Request(
            show_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            parsed = json.loads(raw or "{}")
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            parsed = {}
        _OLLAMA_SHOW_CACHE[cache_key] = parsed
        return parsed

    def estimate_kv_cache_megabytes(
        self,
        base_url: str,
        model_name: str,
        context_window: int,
        used_tokens: int,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if context_window <= 0 or used_tokens < 0:
            return None, None, None

        show_data = self.get_ollama_model_info_cached(base_url, model_name)
        model_info = show_data.get("model_info") if isinstance(show_data.get("model_info"), dict) else {}
        if not isinstance(model_info, dict):
            return None, None, None

        layers = self._pick_int_by_key_fragments(model_info, ["block", "count"]) or self._pick_int_by_key_fragments(model_info, ["layer", "count"])
        hidden_size = self._pick_int_by_key_fragments(model_info, ["embedding", "length"]) or self._pick_int_by_key_fragments(model_info, ["hidden", "size"])
        head_count = self._pick_int_by_key_fragments(model_info, ["attention", "head_count"]) or self._pick_int_by_key_fragments(model_info, ["attn", "head_count"])
        head_count_kv = self._pick_int_by_key_fragments(model_info, ["attention", "head_count_kv"]) or head_count

        if not layers or not hidden_size or not head_count or not head_count_kv:
            return None, None, None
        if head_count <= 0 or hidden_size <= 0:
            return None, None, None

        head_dim = hidden_size // head_count
        if head_dim <= 0:
            return None, None, None

        # Approximation: fp16 KV cache => 2 bytes, K+V => x2.
        bytes_per_token = layers * head_count_kv * head_dim * 2 * 2
        total_bytes = context_window * bytes_per_token
        used_bytes = min(max(used_tokens, 0), context_window) * bytes_per_token
        remaining_bytes = max(total_bytes - used_bytes, 0)
        mb_div = 1024.0 * 1024.0
        return used_bytes / mb_div, remaining_bytes / mb_div, total_bytes / mb_div

    def select_relevant_memory_text(self, source_text: str, query: str, max_chars: int) -> str:
        raw = str(source_text or "").strip()
        if not raw:
            return ""
        blocks = [block.strip() for block in re.split(r"\n\s*\n+", raw) if block.strip()]
        if not blocks:
            return ""
        query_tokens = self.tokenize_query(query)
        ranked: list[tuple[int, str]] = []
        for block in blocks:
            lowered = block.lower()
            score = 0
            for token in query_tokens:
                if token in lowered:
                    score += 2
            if query and query.strip() and query.strip().lower() in lowered:
                score += 3
            if score == 0:
                score = 1
            ranked.append((score, block))
        ranked.sort(key=lambda item: (-item[0], len(item[1])))
        selected: list[str] = []
        used = 0
        for _, block in ranked:
            if used >= max_chars:
                break
            remaining = max_chars - used
            snippet = block if len(block) <= remaining else block[:remaining]
            if not snippet.strip():
                continue
            selected.append(snippet.strip())
            used += len(snippet) + 2
            if len(selected) >= 4:
                break
        return "\n\n".join(selected).strip()

    def build_ollama_injection_prompt(self, user_message: str) -> str:
        once_limit, always_limit, memory_limit = self.get_injection_limits()
        parts: list[str] = []

        if self.is_first_turn_in_session():
            agents_text = self.read_workspace_text("AGENTS.md", once_limit)
            soul_text = self.read_workspace_text("SOUL.md", once_limit)
            identity_text = self.read_workspace_text("IDENTITY.md", once_limit)
            once_chunks: list[str] = []
            if agents_text:
                once_chunks.append(f"[AGENTS]\n{agents_text}")
            if soul_text:
                once_chunks.append(f"[SOUL]\n{soul_text}")
            if identity_text:
                once_chunks.append(f"[IDENTITY]\n{identity_text}")
            if once_chunks:
                parts.append(
                    "Session bootstrap context (apply once at session start; do not quote raw blocks):\n"
                    + "\n\n".join(once_chunks)
                )

        user_text = self.read_workspace_text("USER.md", always_limit)
        if user_text:
            parts.append(f"[USER PREFERENCES]\n{user_text}")

        long_memory_text = self.read_workspace_text("MEMORY.md", memory_limit * 3)
        long_memory_selected = self.select_relevant_memory_text(
            long_memory_text,
            user_message,
            memory_limit,
        )
        if long_memory_selected:
            parts.append(f"[LONG-TERM MEMORY]\n{long_memory_selected}")

        daily_path = self.get_profile_workspace_dir() / "memory" / f"{date.today().isoformat()}.md"
        daily_text = ""
        try:
            if daily_path.exists():
                daily_text = daily_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            daily_text = ""
        daily_selected = self.select_relevant_memory_text(
            daily_text,
            user_message,
            memory_limit,
        )
        if daily_selected:
            parts.append(f"[DAILY MEMORY]\n{daily_selected}")

        if not parts:
            return ""
        return (
            "Use the following profile context and memory as soft guidance. "
            "Do not print labels like AGENTS/SOUL/IDENTITY/USER/MEMORY in final answer.\n\n"
            + "\n\n".join(parts)
        )

    def build_direct_ollama_messages(
        self,
        user_message: str,
        extra_system_prompt: Optional[str] = None,
    ) -> list[dict]:
        system_prompt = "This is an interactive user chat. Always answer the user directly. Do NOT reply with HEARTBEAT_OK."
        if extra_system_prompt:
            system_prompt = f"{system_prompt}\n{extra_system_prompt}"
        injection = self.build_ollama_injection_prompt(user_message)
        if injection:
            system_prompt = f"{system_prompt}\n\n{injection}"

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        history_items = [
            {
                "role": str(item.get("role") or "").strip(),
                "content": str(item.get("content") or "").strip(),
            }
            for item in self.history
            if isinstance(item, dict)
            and str(item.get("role") or "").strip() in {"user", "assistant"}
            and str(item.get("content") or "").strip()
        ]
        if history_items and history_items[-1]["role"] == "user" and history_items[-1]["content"] == str(user_message or "").strip():
            history_items = history_items[:-1]
        history_turns = self.get_ollama_history_turns()
        history_items = history_items[-(history_turns * 2):]
        messages.extend(history_items)
        messages.append({"role": "user", "content": user_message})
        return messages

    def persist_daily_memory_note(self, user_message: str, assistant_message: str) -> None:
        user_text = str(user_message or "").strip()
        assistant_text = str(assistant_message or "").strip()
        if not user_text or not assistant_text:
            return
        try:
            memory_dir = self.get_profile_workspace_dir() / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
            daily_file = memory_dir / f"{date.today().isoformat()}.md"
            if not daily_file.exists():
                daily_file.write_text(
                    f"# Daily Memory {date.today().isoformat()}\n\n",
                    encoding="utf-8",
                )
            timestamp = datetime.now().strftime("%H:%M")
            user_short = user_text if len(user_text) <= 420 else f"{user_text[:420]}..."
            assistant_short = assistant_text if len(assistant_text) <= 680 else f"{assistant_text[:680]}..."
            entry = (
                f"## {timestamp}\n"
                f"- user: {user_short}\n"
                f"- assistant: {assistant_short}\n\n"
            )
            with daily_file.open("a", encoding="utf-8") as fp:
                fp.write(entry)
        except Exception as e:
            self.debug_received.emit(f"daily memory append 실패: {e}")

    def send_via_webchat_ws(
        self,
        token: str,
        session_key: str,
        user_message: str,
        extra_system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        ws = websocket.create_connection("ws://127.0.0.1:18789", timeout=10)
        ws.settimeout(60)
        try:
            # Wait for connect.challenge (optional)
            for _ in range(5):
                msg = ws.recv()
                try:
                    obj = json.loads(msg)
                except Exception:
                    continue
                if obj.get("type") == "event" and obj.get("event") == "connect.challenge":
                    break

            connect_id = str(uuid.uuid4())
            connect_req = {
                "type": "req",
                "id": connect_id,
                "method": "connect",
                "params": {
                    "minProtocol": 3,
                    "maxProtocol": 3,
                    "client": {
                        "id": "webchat-ui",
                        "displayName": "GhostRelay",
                        "version": "1.0.0",
                        "platform": "windows",
                        "mode": "webchat",
                        "instanceId": str(uuid.uuid4()),
                    },
                    "role": "operator",
                    "scopes": [],
                    "auth": {"token": token},
                },
            }
            ws.send(json.dumps(connect_req))

            connected = False
            for _ in range(10):
                msg = ws.recv()
                obj = json.loads(msg)
                if obj.get("type") == "res" and obj.get("id") == connect_id:
                    if not obj.get("ok"):
                        raise RuntimeError(obj.get("error", {}).get("message", "connect failed"))
                    connected = True
                    break
            if not connected:
                raise RuntimeError("connect timeout")

            send_id = str(uuid.uuid4())
            run_id = str(uuid.uuid4())
            agent_req = {
                "type": "req",
                "id": send_id,
                "method": "agent",
                "params": {
                    "agentId": self.agent_id,
                    "sessionKey": session_key,
                    "message": user_message,
                    "idempotencyKey": run_id,
                    "thinking": self.thinking_mode,
                    "extraSystemPrompt": (
                        extra_system_prompt
                        or "This is an interactive user chat. "
                        "Always answer the user directly. Do NOT reply with HEARTBEAT_OK."
                    ),
                },
            }
            ws.send(json.dumps(agent_req))

            response_text = None
            accepted_run_id = None
            stream_text = ""
            last_partial_emit = 0.0
            last_partial_text = ""

            for _ in range(600):
                msg = ws.recv()
                obj = json.loads(msg)
                if obj.get("type") == "event":
                    event_name = obj.get("event")
                    payload = obj.get("payload", {}) or {}
                    payload_run_id = payload.get("runId")
                    if accepted_run_id and payload_run_id and payload_run_id != accepted_run_id:
                        continue

                    # agent 이벤트 스트리밍
                    if event_name == "agent":
                        data = payload.get("data", {}) or {}
                        chunk = data.get("text")
                        if isinstance(chunk, str) and chunk:
                            # 일부 런타임은 누적 텍스트, 일부는 델타 텍스트를 보냄
                            if stream_text and chunk.startswith(stream_text):
                                stream_text = chunk
                            elif len(chunk) >= len(stream_text) and stream_text and stream_text in chunk:
                                stream_text = chunk
                            else:
                                stream_text += chunk
                            preview = stream_text.strip()
                            now = time.time()
                            should_emit = (
                                preview
                                and preview != last_partial_text
                                and (now - last_partial_emit >= 0.18)
                            )
                            if should_emit and not self.is_heartbeat_like(preview):
                                self.partial_received.emit(preview)
                                last_partial_text = preview
                                last_partial_emit = now
                        continue

                    # chat delta/final 이벤트도 보조로 처리
                    if event_name == "chat":
                        state = payload.get("state")
                        if state in {"delta", "final"}:
                            message = payload.get("message") or {}
                            content = message.get("content") or []
                            if isinstance(content, list):
                                text_parts = []
                                for item in content:
                                    if isinstance(item, dict) and item.get("text"):
                                        text_parts.append(str(item.get("text")))
                                if text_parts:
                                    joined = "\n".join(text_parts).strip()
                                    if joined:
                                        stream_text = joined
                                        now = time.time()
                                        should_emit = (
                                            stream_text
                                            and stream_text != last_partial_text
                                            and (now - last_partial_emit >= 0.18)
                                        )
                                        if should_emit and not self.is_heartbeat_like(stream_text):
                                            self.partial_received.emit(stream_text)
                                            last_partial_text = stream_text
                                            last_partial_emit = now
                        continue

                if obj.get("type") == "res" and obj.get("id") == send_id:
                    if not obj.get("ok"):
                        raise RuntimeError(obj.get("error", {}).get("message", "agent error"))
                    payload = obj.get("payload", {})
                    status = payload.get("status")
                    if status == "accepted":
                        accepted_run_id = payload.get("runId")
                        continue
                    if status == "error":
                        summary = str(payload.get("summary") or payload.get("error") or "agent error")
                        raise RuntimeError(summary)
                    if status != "ok":
                        continue
                    result = payload.get("result") or {}
                    payloads = result.get("payloads") or []
                    texts = []
                    if isinstance(payloads, list):
                        for item in payloads:
                            if isinstance(item, dict) and item.get("text"):
                                texts.append(str(item.get("text")))
                    response_text = "\n".join(texts).strip() if texts else None
                    extracted = self.extract_text_from_payload_blob(response_text or "")
                    if extracted:
                        response_text = extracted
                    if self.is_provider_error_text(response_text):
                        raise RuntimeError(response_text or "provider error")
                    break
            return response_text
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def send_via_openai_http(
        self,
        token: str,
        session_key: str,
        user_message: str,
        extra_system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        import urllib.request
        import urllib.error

        payload = {
            "model": "openclaw:main",
            "messages": self.build_openai_messages(user_message, extra_system_prompt),
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:18789/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "x-openclaw-session-key": session_key,
                "x-openclaw-agent-id": self.agent_id,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = str(e)
            raise RuntimeError(f"HTTP {e.code}: {body}")
        obj = json.loads(raw)
        choices = obj.get("choices") or []
        if choices and isinstance(choices, list):
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
        return None

    def send_via_direct_ollama_http(
        self,
        user_message: str,
        extra_system_prompt: Optional[str] = None,
    ) -> tuple[Optional[str], dict[str, Any]]:
        import urllib.request
        import urllib.error

        settings = self.profile_settings if isinstance(self.profile_settings, dict) else {}
        ollama = settings.get("ollama", {}) if isinstance(settings.get("ollama"), dict) else {}
        base_url = str(ollama.get("baseUrl") or "http://127.0.0.1:11434/v1").strip()
        api_key = str(ollama.get("apiKey") or "ollama-local").strip() or "ollama-local"
        timeout_sec = clamp_int(
            ollama.get("requestTimeoutSec"),
            DEFAULT_OLLAMA_HTTP_TIMEOUT_SEC,
            30,
            600,
        )
        model_name = str(ollama.get("modelName") or "").strip()
        if not model_name:
            model_ref = str(settings.get("model") or "").strip()
            if model_ref.startswith("ollama/"):
                model_name = model_ref.split("/", 1)[1].strip()
        if not model_name:
            model_name = "llama3.3"

        raw_base = base_url.rstrip("/")
        if not raw_base:
            raw_base = "http://127.0.0.1:11434/v1"
        if raw_base.endswith("/v1"):
            endpoint = f"{raw_base}/chat/completions"
        else:
            endpoint = f"{raw_base}/v1/chat/completions"

        messages = self.build_direct_ollama_messages(user_message, extra_system_prompt)
        estimated_prompt_tokens = 0
        for item in messages:
            if isinstance(item, dict):
                estimated_prompt_tokens += self.estimate_token_count(str(item.get("content") or ""))
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers=headers,
            method="POST",
        )
        request_started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = str(e)
            raise RuntimeError(f"HTTP {e.code}: {body}")
        elapsed_sec = max(time.time() - request_started, 0.001)
        obj = json.loads(raw or "{}")
        choices = obj.get("choices") or []
        response_text: Optional[str] = None
        if choices and isinstance(choices, list):
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                response_text = content.strip()

        usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
        prompt_tokens = clamp_int(
            usage.get("prompt_tokens"),
            estimated_prompt_tokens,
            0,
            2_000_000,
        )
        completion_tokens = clamp_int(
            usage.get("completion_tokens"),
            self.estimate_token_count(response_text or ""),
            0,
            2_000_000,
        )
        total_tokens = clamp_int(
            usage.get("total_tokens"),
            prompt_tokens + completion_tokens,
            0,
            2_000_000,
        )
        context_window = clamp_int(
            ollama.get("contextWindow"),
            32768,
            1,
            2_000_000,
        )
        context_used = min(total_tokens, context_window)
        context_remaining = max(context_window - context_used, 0)
        tokens_per_sec = float(completion_tokens) / elapsed_sec if completion_tokens > 0 else 0.0
        kv_used_mb, kv_remaining_mb, kv_total_mb = self.estimate_kv_cache_megabytes(
            base_url,
            model_name,
            context_window,
            context_used,
        )
        stats = {
            "provider": "ollama",
            "model": model_name,
            "contextWindow": context_window,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": total_tokens,
            "contextUsed": context_used,
            "contextRemaining": context_remaining,
            "contextUsagePct": (float(context_used) / float(context_window) * 100.0) if context_window > 0 else 0.0,
            "latencySec": elapsed_sec,
            "tokensPerSec": tokens_per_sec,
            "kvUsedMB": kv_used_mb,
            "kvRemainingMB": kv_remaining_mb,
            "kvTotalMB": kv_total_mb,
        }
        return response_text, stats
        
    def run(self):
        """Agent 실행"""
        try:
            started_at = time.time()
            openclaw_mjs = self.openclaw_dir / "openclaw.mjs"
            if not openclaw_mjs.exists():
                self.message_received.emit("❌ OpenClaw CLI를 찾을 수 없습니다")
                return
            
            # Agent 실행 (Gateway를 통해)
            # --to 옵션은 채널이 필요하므로, Gateway WebSocket을 직접 사용하거나
            # --local 옵션을 사용할 수 있음
            # 일단 Gateway를 통해 실행 (기본 세션 사용)
            env = os.environ.copy()
            env["NRG_OPENCLAW_AGENT_LOCAL"] = "0"  # Gateway 사용
            env["OPENCLAW_GATEWAY_PORT"] = "18789"
            if self.gateway_token:
                env["OPENCLAW_GATEWAY_TOKEN"] = self.gateway_token
            try:
                settings = self.profile_settings or load_settings()
                vertex = settings.get("vertex", {}) if isinstance(settings, dict) else {}
                sa_path = str(vertex.get("serviceAccountPath") or "").strip()
                project = str(vertex.get("project") or "").strip()
                location = str(vertex.get("location") or "").strip()
                if sa_path:
                    env["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
                if project:
                    env["GOOGLE_CLOUD_PROJECT"] = project
                    env["GCLOUD_PROJECT"] = project
                if location:
                    env["GOOGLE_CLOUD_LOCATION"] = location
            except Exception:
                pass
            
            session_key = self.resolve_session_key()
            first_prompt = (
                "This is an interactive user chat. Always answer the user directly. "
                "Do NOT reply with HEARTBEAT_OK."
            )
            retry_prompt = (
                "This is an interactive user chat. The user input may be short or ambiguous. "
                "Never output HEARTBEAT_OK. If ambiguous, ask one concise clarifying question in Korean."
            )
            dispatch_mode = self.resolve_dispatch_mode()
            source_is_ollama = self.get_profile_model_source() == "ollama"
            token = (self.gateway_token or "").strip()
            requires_gateway_token = not (dispatch_mode == "chat" and source_is_ollama)
            if requires_gateway_token and not token:
                raise RuntimeError("Missing gateway auth token. Check gateway.auth.token.")
            if dispatch_mode == "chat":
                self.debug_received.emit(
                    f"대화 전용 모드(HTTP) 연결 시도 (agentId={self.agent_id}, sessionKey={session_key})"
                )
            else:
                self.debug_received.emit(f"WS 연결 시도 (agentId={self.agent_id}, sessionKey={session_key})")
            initial_message = self.user_input
            if dispatch_mode == "chat":
                use_direct_ollama = source_is_ollama
                try:
                    if use_direct_ollama:
                        response_text, direct_stats = self.send_via_direct_ollama_http(
                            initial_message,
                            first_prompt,
                        )
                        if isinstance(direct_stats, dict):
                            self.stats_received.emit(direct_stats)
                    else:
                        response_text = self.send_via_openai_http(
                            token,
                            session_key,
                            initial_message,
                            first_prompt,
                        )
                except RuntimeError as http_err:
                    http_err_text = str(http_err)
                    if self.is_provider_rate_limit_error(http_err_text):
                        elapsed = time.time() - started_at
                        self.message_received.emit(
                            "Vertex API 한도 초과(429)입니다. 잠시 후 다시 시도하거나 "
                            "다른 모델로 바꿔주세요.\n\n"
                            f"원인: {http_err_text}\n\n(응답 {elapsed:.1f}초)"
                        )
                        return
                    raise

                if not self.is_heartbeat_like(response_text):
                    response_text = self.repair_chat_response(
                        response_text,
                        token,
                        session_key,
                        use_direct_ollama,
                    )
                    if use_direct_ollama:
                        self.persist_daily_memory_note(self.user_input, response_text)
                    elapsed = time.time() - started_at
                    self.message_received.emit(f"{response_text}\n\n(응답 {elapsed:.1f}초)")
                    return

                retry_message = self.build_retry_message()
                self.debug_received.emit("HTTP 1차 응답이 heartbeat/비정상. HTTP 2차 재시도...")
                if use_direct_ollama:
                    response_text, retry_stats = self.send_via_direct_ollama_http(
                        retry_message,
                        retry_prompt,
                    )
                    if isinstance(retry_stats, dict):
                        self.stats_received.emit(retry_stats)
                else:
                    response_text = self.send_via_openai_http(
                        token,
                        session_key,
                        retry_message,
                        retry_prompt,
                    )
                if not self.is_heartbeat_like(response_text):
                    response_text = self.repair_chat_response(
                        response_text,
                        token,
                        session_key,
                        use_direct_ollama,
                    )
                    if use_direct_ollama:
                        self.persist_daily_memory_note(self.user_input, response_text)
                    elapsed = time.time() - started_at
                    self.message_received.emit(f"{response_text}\n\n(응답 {elapsed:.1f}초)")
                    return
                self.message_received.emit(
                    "입력이 너무 짧거나 모호합니다. 한 문장으로 다시 입력해 주세요."
                )
                return
            try:
                response_text = self.send_via_webchat_ws(
                    token,
                    session_key,
                    initial_message,
                    first_prompt,
                )
            except RuntimeError as ws_err:
                ws_err_text = str(ws_err)
                if self.is_provider_rate_limit_error(ws_err_text):
                    elapsed = time.time() - started_at
                    self.message_received.emit(
                        "Vertex API 한도 초과(429)입니다. 잠시 후 다시 시도하거나 "
                        "다른 모델로 바꿔주세요.\n\n"
                        f"원인: {ws_err_text}\n\n(응답 {elapsed:.1f}초)"
                    )
                    return
                if self.is_tools_unsupported_error(ws_err_text):
                    self.debug_received.emit("모델이 tools 호출을 지원하지 않아 HTTP(OpenAI 호환) 폴백을 시도합니다.")
                    try:
                        response_text = self.send_via_openai_http(
                            token,
                            session_key,
                            initial_message,
                            first_prompt,
                        )
                    except RuntimeError as http_err:
                        http_err_text = str(http_err)
                        if self.is_provider_rate_limit_error(http_err_text):
                            elapsed = time.time() - started_at
                            self.message_received.emit(
                                "Vertex API 한도 초과(429)입니다. 잠시 후 다시 시도하거나 "
                                "다른 모델로 바꿔주세요.\n\n"
                                f"원인: {http_err_text}\n\n(응답 {elapsed:.1f}초)"
                            )
                            return
                        self.debug_received.emit("HTTP 폴백 실패. provider 무툴 정책을 적용해 WS 재시도합니다.")
                        policy_ok, policy_msg = self.apply_provider_no_tools_policy()
                        if policy_msg:
                            self.debug_received.emit(policy_msg)
                        if not policy_ok:
                            raise RuntimeError(
                                f"{ws_err_text}\n\n자동 무툴 정책 적용 실패: {policy_msg}"
                            )
                        response_text = self.send_via_webchat_ws(
                            token,
                            session_key,
                            initial_message,
                            first_prompt,
                        )
                else:
                    raise

            if not self.is_heartbeat_like(response_text):
                response_text = self.clean_response_text(response_text) or (response_text or "").strip()
                if self.is_tool_call_like(response_text):
                    self.debug_received.emit("도구 호출 형태 응답 감지. 자연어 답변으로 재요청...")
                    natural_prompt = (
                        "You are chatting with a normal user. "
                        "Do not output tool-call syntax. "
                        "Return a plain Korean sentence response only."
                    )
                    natural_message = (
                        f'사용자 원문: "{self.user_input}"\n'
                        f'방금 내부 출력: "{response_text}"\n'
                        "도구 호출문을 노출하지 말고 한국어 자연어 답변만 해주세요."
                    )
                    refined = self.send_via_webchat_ws(
                        token,
                        session_key,
                        natural_message,
                        natural_prompt,
                    )
                    if refined and not self.is_heartbeat_like(refined):
                        response_text = refined
                if self.is_tool_call_like(response_text):
                    response_text = (
                        "요청 내용을 이해했어요. 일반 대화 형태로 답하도록 재시도해 주세요. "
                        "가능하면 작업 목적을 한 문장으로 조금 더 구체적으로 적어주세요."
                    )
                elapsed = time.time() - started_at
                self.message_received.emit(f"{response_text}\n\n(응답 {elapsed:.1f}초)")
                return

            retry_message = self.build_retry_message()
            self.debug_received.emit("WS 1차 응답이 heartbeat/비정상. WS 2차 강제 재시도...")
            response_text = self.send_via_webchat_ws(
                token,
                session_key,
                retry_message,
                retry_prompt,
            )
            if not self.is_heartbeat_like(response_text):
                response_text = self.clean_response_text(response_text) or (response_text or "").strip()
                elapsed = time.time() - started_at
                self.message_received.emit(f"{response_text}\n\n(응답 {elapsed:.1f}초)")
                return

            self.debug_received.emit("WS 2차도 heartbeat/비정상. HTTP(OpenAI 호환) 폴백...")
            try:
                response_text = self.send_via_openai_http(
                    token,
                    session_key,
                    retry_message,
                    retry_prompt,
                )
            except RuntimeError as http_err:
                http_err_text = str(http_err)
                if self.is_provider_rate_limit_error(http_err_text):
                    elapsed = time.time() - started_at
                    self.message_received.emit(
                        "Vertex API 한도 초과(429)입니다. 잠시 후 다시 시도하거나 "
                        "다른 모델로 바꿔주세요.\n\n"
                        f"원인: {http_err_text}\n\n(응답 {elapsed:.1f}초)"
                    )
                    return
                raise
            if not self.is_heartbeat_like(response_text):
                response_text = self.clean_response_text(response_text) or (response_text or "").strip()
                elapsed = time.time() - started_at
                self.message_received.emit(f"{response_text}\n\n(응답 {elapsed:.1f}초)")
                return

            self.message_received.emit(
                "입력이 너무 짧거나 모호합니다. 한 문장으로 다시 입력해 주세요."
            )
                
        except subprocess.TimeoutExpired:
            self.message_received.emit("⏱️ 응답 시간 초과 (60초)")
        except TimeoutError:
            self.message_received.emit(
                "⏱️ 모델 응답 대기 시간이 초과되었습니다. "
                "첫 호출에서 모델 로딩 중일 수 있으니 잠시 후 다시 시도해 주세요."
            )
        except Exception as e:
            self.message_received.emit(f"❌ 오류: {str(e)}")


class GhostChatWindow(QMainWindow):
    """고스트 챗 메인 윈도우 (BRAIN 전용)"""
    
    def __init__(
        self,
        profile_id: Optional[str] = None,
        use_tray_icon: bool = True,
        console_managed: bool = False,
    ):
        super().__init__()
        self.gateway_manager = OpenClawGatewayManager()
        self.use_tray_icon = bool(use_tray_icon)
        self.console_managed = bool(console_managed)
        self.settings_cache = load_settings()
        self.profile_id = normalize_profile_id(
            profile_id if profile_id is not None else get_active_profile_id(self.settings_cache)
        )
        self.profile_settings = get_profile(self.settings_cache, self.profile_id)
        self.ui_language = self.resolve_ui_language(self.settings_cache)
        self.runtime_session_key = generate_runtime_session_key()
        self.startup_thread: Optional[GatewayStartupThread] = None
        self.agent_thread: Optional[OpenClawAgentThread] = None
        self.chat_history: list[dict] = []
        self.processing_started_at: Optional[float] = None
        self.processing_step = 0
        self.stream_preview_active = False
        self.stream_preview_start_pos: Optional[int] = None
        self.last_partial_preview = ""
        self.last_partial_preview_at = 0.0
        self.hidden_reasoning_store: dict[str, str] = {}
        self.hidden_reasoning_counter = 0
        self.last_runtime_stats: dict[str, Any] = {}
        self.drag_position = QPoint()
        self._is_dragging = False
        self._is_resizing = False
        self._resize_edges = {"left": False, "right": False, "top": False, "bottom": False}
        self._resize_start_geom = QRect()
        self._resize_start_pos = QPoint()
        self._resize_margin = 8
        self._header_drag_height = 68
        self._force_close = False
        self.always_on_top = bool(self.settings_cache.get("alwaysOnTop", True))
        self.editor_file_paths: dict[str, Path] = {}
        self.editor_path_labels: dict[str, QLabel] = {}
        self.editor_widgets: dict[str, QTextEdit] = {}
        self.editor_file_signatures: dict[str, Optional[tuple[int, int]]] = {}
        self.pending_external_refresh_ids: set[str] = set()
        self.file_watch_paths: set[str] = set()
        self.file_watcher = QFileSystemWatcher(self)
        self.file_watcher.fileChanged.connect(self.on_watched_file_changed)
        self.file_watcher.directoryChanged.connect(self.on_watched_directory_changed)
        self.external_refresh_timer = QTimer(self)
        self.external_refresh_timer.setSingleShot(True)
        self.external_refresh_timer.setInterval(300)
        self.external_refresh_timer.timeout.connect(self.flush_pending_external_refresh)
        ui_settings = self.settings_cache.get("ui", {}) if isinstance(self.settings_cache.get("ui"), dict) else {}
        self.left_panel_visible = bool(ui_settings.get("leftPanelVisible", False))
        self.right_panel_visible = bool(ui_settings.get("rightPanelVisible", True))
        self.profile_sync_thread: Optional[ProfileRuntimeSyncThread] = None
        self.profile_sync_pending = False
        self.init_ui()
        self.refresh_editor_watch_targets()
        self.refresh_profile_selector()
        self.update_window_title()
        if self.use_tray_icon:
            self.init_tray()
        else:
            self.tray = None
        self.sync_always_on_top_controls()
        self.startup_sequence()

    def resolve_ui_language(self, settings: dict | None = None) -> str:
        source = settings if isinstance(settings, dict) else self.settings_cache
        ui_settings = source.get("ui", {}) if isinstance(source.get("ui"), dict) else {}
        lang = str(ui_settings.get("language") or "ko").strip().lower()
        return lang if lang in {"ko", "en"} else "ko"

    def tr(self, ko: str, en: str) -> str:
        return en if self.ui_language == "en" else ko

    def update_window_title(self):
        profile = normalize_profile_id(self.profile_id)
        self.setWindowTitle(f"👻 GhostRelay Chat [{profile}]")
        if hasattr(self, "title_label"):
            self.title_label.setText(f"👻 Ghost Chat [{profile}]")

    def refresh_profile_selector(self):
        if not hasattr(self, "profile_combo"):
            return
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for profile in list_profiles(self.settings_cache):
            self.profile_combo.addItem(profile, profile)
        idx = self.profile_combo.findData(self.profile_id)
        if idx < 0 and self.profile_combo.count() > 0:
            idx = 0
        if idx >= 0:
            self.profile_combo.setCurrentIndex(idx)
        self.profile_combo.blockSignals(False)

    def reload_profile_runtime(self, profile_id: str, announce: bool = False):
        target = normalize_profile_id(profile_id)
        self.profile_id = target
        self.refresh_settings_cache()
        self.runtime_session_key = generate_runtime_session_key()
        self.chat_history.clear()
        self.last_partial_preview = ""
        self.last_partial_preview_at = 0.0
        self.last_runtime_stats = {}
        if hasattr(self, "runtime_stats_label"):
            self.runtime_stats_label.setText("Ctx - | KV - | tok/s -")
            self.runtime_stats_label.setToolTip("런타임 지표가 여기에 표시됩니다.")
        self.update_window_title()
        self.refresh_profile_selector()
        self.update_editor_targets_for_profile()
        if hasattr(self, "session_key_input"):
            self.session_key_input.setText(str(self.profile_settings.get("sessionKey") or "").strip())
        self.gateway_manager.settings = self.profile_settings
        self.request_profile_runtime_sync(announce=False)
        if announce:
            self.append_system_line(
                f"✅ 프로필 전환: <b>{html.escape(self.profile_id)}</b> "
                "(세션 컨텍스트를 새로 시작합니다)"
            )

    def request_profile_runtime_sync(self, announce: bool = False):
        self.gateway_manager.settings = self.profile_settings
        if self.profile_sync_thread and self.profile_sync_thread.isRunning():
            self.profile_sync_pending = True
            return
        self.profile_sync_pending = False
        profile_snapshot = dict(self.profile_settings) if isinstance(self.profile_settings, dict) else {}
        self.profile_sync_thread = ProfileRuntimeSyncThread(profile_snapshot)
        self.profile_sync_thread.completed.connect(self.on_profile_runtime_sync_completed)
        self.profile_sync_thread.start()
        if announce:
            self.append_debug("프로필 런타임 동기화 시작...")

    def on_profile_runtime_sync_completed(self, ok: bool, details_obj: object, restart_hint: bool):
        details = [str(item) for item in details_obj] if isinstance(details_obj, list) else []
        if ok:
            if details:
                self.append_debug(" / ".join(details))
        else:
            self.append_system_line("⚠️ 프로필 런타임 동기화 중 일부 실패가 발생했습니다.")
            if details:
                self.append_system_line("<br>".join(html.escape(item) for item in details))

        if restart_hint and self.gateway_manager.check_gateway_running():
            self.append_system_line(
                "ℹ️ 프로필 설정이 갱신되었습니다. "
                "워크스페이스/모델 반영을 위해 게이트웨이를 재시작하면 더 안정적입니다."
            )

        self.profile_sync_thread = None
        if self.profile_sync_pending:
            self.profile_sync_pending = False
            self.request_profile_runtime_sync(announce=False)

    def on_profile_combo_changed(self):
        selected = str(self.profile_combo.currentData() or "").strip()
        if not selected:
            return
        if normalize_profile_id(selected) == normalize_profile_id(self.profile_id):
            return
        if self.agent_thread and self.agent_thread.isRunning():
            self.append_system_line("⏳ 응답 처리 중에는 프로필을 변경할 수 없습니다.")
            self.refresh_profile_selector()
            return
        self.reload_profile_runtime(selected, announce=True)

    def open_embedded_settings(self):
        try:
            from launcher import SettingsDialog
        except Exception as e:
            self.append_system_line(f"❌ 설정창 로드 실패: {html.escape(str(e))}")
            return
        dialog = SettingsDialog(self, initial_profile=self.profile_id)
        if dialog.exec() == QDialog.Accepted:
            next_profile = normalize_profile_id(getattr(dialog, "active_profile_id", self.profile_id))
            self.reload_profile_runtime(next_profile, announce=True)

    def init_ui(self):
        self.setWindowTitle("👻 GhostRelay Chat")
        self.setMinimumSize(760, 760)
        self.resize(720, 820)
        flags = Qt.FramelessWindowHint | Qt.Tool
        if self.always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 중앙 위젯
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 헤더 (드래그 영역)
        header = QWidget()
        header.setFixedHeight(self._header_drag_height)
        header.setStyleSheet("""
            QWidget {
                background: rgba(21, 23, 28, 0.95);
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
            }
        """)
        self.header = header
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(10, 6, 10, 6)
        header_layout.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)

        title = QLabel("👻 Ghost Chat")
        title.setStyleSheet("color: white; font-weight: 600; font-size: 14px;")
        self.title_label = title
        top_row.addWidget(title)

        self.profile_combo = QComboBox()
        self.profile_combo.setFixedWidth(120)
        self.profile_combo.setToolTip(self.tr("이 창의 프로필", "Profile for this window"))
        self.profile_combo.setStyleSheet("""
            QComboBox {
                background: rgba(14, 15, 18, 0.85);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 6px;
                padding: 2px 6px;
                font-size: 10px;
            }
            QComboBox::drop-down {
                border: none;
                width: 16px;
            }
            QComboBox QAbstractItemView {
                background: #11151d;
                color: white;
                border: 1px solid #2a3341;
                selection-background-color: #2a3341;
            }
        """)
        self.profile_combo.currentIndexChanged.connect(self.on_profile_combo_changed)
        top_row.addWidget(self.profile_combo)

        self.settings_button = QPushButton(self.tr("설정", "Settings"))
        self.settings_button.setFixedHeight(22)
        self.settings_button.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.10);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 4px;
                padding: 0 8px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.18);
            }
        """)
        self.settings_button.clicked.connect(self.open_embedded_settings)
        top_row.addWidget(self.settings_button)

        top_row.addStretch()

        # 상태 표시
        self.status_label = QLabel("준비 중...")
        self.status_label.setStyleSheet("color: rgba(255, 255, 255, 0.68); font-size: 11px;")
        top_row.addWidget(self.status_label)

        self.runtime_stats_label = QLabel("Ctx - | KV - | tok/s -")
        self.runtime_stats_label.setStyleSheet("color: rgba(140, 198, 255, 0.88); font-size: 10px;")
        self.runtime_stats_label.setToolTip("런타임 지표가 여기에 표시됩니다.")
        top_row.addWidget(self.runtime_stats_label)

        self.options_button = QPushButton(self.tr("옵션", "Options"))
        self.options_button.setFixedHeight(22)
        self.options_button.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.10);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 4px;
                padding: 0 8px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.18);
            }
        """)
        self.options_button.clicked.connect(self.show_options_menu)
        top_row.addWidget(self.options_button)

        # 최소화 버튼
        btn_minimize = QPushButton("−")
        btn_minimize.setFixedSize(22, 22)
        btn_minimize.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.1);
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.2);
            }
        """)
        btn_minimize.clicked.connect(self.hide)
        top_row.addWidget(btn_minimize)

        # 닫기 버튼
        btn_close = QPushButton("×")
        btn_close.setFixedSize(22, 22)
        btn_close.setStyleSheet("""
            QPushButton {
                background: rgba(255, 90, 90, 0.2);
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: rgba(255, 90, 90, 0.35);
            }
        """)
        btn_close.clicked.connect(self.close)
        top_row.addWidget(btn_close)

        control_row = QHBoxLayout()
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(8)

        # 디버그 토글
        self.debug_toggle = QCheckBox(self.tr("디버그", "Debug"))
        self.debug_toggle.setStyleSheet("color: rgba(255, 255, 255, 0.68); font-size: 10px;")
        control_row.addWidget(self.debug_toggle)

        self.always_on_top_toggle = QCheckBox(self.tr("항상위", "Topmost"))
        self.always_on_top_toggle.setChecked(self.always_on_top)
        self.always_on_top_toggle.setToolTip("창을 항상 최상단에 고정")
        self.always_on_top_toggle.setStyleSheet("color: rgba(255, 255, 255, 0.68); font-size: 10px;")
        self.always_on_top_toggle.toggled.connect(self.on_always_on_top_toggled)
        control_row.addWidget(self.always_on_top_toggle)

        # thinking 레벨 선택
        thinking_label = QLabel("Think")
        thinking_label.setStyleSheet("color: rgba(255, 255, 255, 0.75); font-size: 10px;")
        control_row.addWidget(thinking_label)

        self.thinking_combo = QComboBox()
        self.thinking_combo.addItems(["off", "minimal", "low", "medium", "high"])
        self.thinking_combo.setCurrentText("off")
        self.thinking_combo.setToolTip("응답 품질/속도 조절: off가 가장 빠름")
        self.thinking_combo.setFixedWidth(94)
        self.thinking_combo.setStyleSheet("""
            QComboBox {
                background: rgba(14, 15, 18, 0.85);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 6px;
                padding: 2px 6px;
                font-size: 10px;
            }
            QComboBox::drop-down {
                border: none;
                width: 16px;
            }
            QComboBox QAbstractItemView {
                background: #11151d;
                color: white;
                border: 1px solid #2a3341;
                selection-background-color: #2a3341;
            }
        """)
        control_row.addWidget(self.thinking_combo)

        self.collapse_reasoning_check = QCheckBox("사고접기")
        self.collapse_reasoning_check.setChecked(True)
        self.collapse_reasoning_check.setToolTip(
            self.tr(
                "중간 사고/도구 출력은 접고 최종 답변만 표시. 접힌 내용은 메시지의 링크로 열람 가능",
                "Collapse intermediate reasoning/tool output. Open folded content via message link.",
            )
        )
        self.collapse_reasoning_check.setStyleSheet("color: rgba(255, 255, 255, 0.75); font-size: 10px;")
        control_row.addWidget(self.collapse_reasoning_check)

        control_row.addStretch()

        # 투명도 슬라이더
        opacity_label = QLabel(self.tr("투명도", "Opacity"))
        opacity_label.setStyleSheet("color: rgba(255, 255, 255, 0.6); font-size: 10px;")
        control_row.addWidget(opacity_label)

        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setMinimum(30)
        opacity_slider.setMaximum(100)
        opacity_slider.setValue(92)
        opacity_slider.setFixedWidth(98)
        opacity_slider.valueChanged.connect(self.update_opacity)
        control_row.addWidget(opacity_slider)
        self.opacity_slider = opacity_slider

        header_layout.addLayout(top_row)
        header_layout.addLayout(control_row)
        layout.addWidget(header)

        self.options_menu = QMenu(self)
        self.left_panel_action = QAction(self.tr("좌측 패널: Persona", "Left Panel: Persona"), self)
        self.left_panel_action.setCheckable(True)
        self.left_panel_action.setChecked(self.left_panel_visible)
        self.left_panel_action.toggled.connect(self.on_toggle_left_panel)
        self.options_menu.addAction(self.left_panel_action)

        self.right_panel_action = QAction(
            self.tr("우측 패널: Memory/Session/Skills", "Right Panel: Memory/Session/Skills"),
            self,
        )
        self.right_panel_action.setCheckable(True)
        self.right_panel_action.setChecked(self.right_panel_visible)
        self.right_panel_action.toggled.connect(self.on_toggle_right_panel)
        self.options_menu.addAction(self.right_panel_action)

        # Zone B: Chat + Side Panels
        center_container = QWidget()
        center_layout = QHBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        targets = self.get_editor_targets()
        left_tabs = [
            (
                "persona_agents",
                "AGENTS",
                targets["persona_agents"],
                self.tr(
                    "시스템 운영 규칙/우선순위 지침을 입력하세요.",
                    "Write system operating rules and priority policy.",
                ),
            ),
            (
                "persona_soul",
                "SOUL",
                targets["persona_soul"],
                self.tr(
                    "말투/스타일/경계(페르소나)를 입력하세요.",
                    "Write tone/style/boundaries (persona).",
                ),
            ),
            (
                "persona_identity",
                "IDENTITY",
                targets["persona_identity"],
                self.tr(
                    "에이전트 정체성/별칭/톤을 입력하세요.",
                    "Write agent identity/alias/tone.",
                ),
            ),
            (
                "persona_user",
                "USER",
                targets["persona_user"],
                self.tr(
                    "사용자 선호/금지사항/호칭을 입력하세요.",
                    "Write user preferences/restrictions/addressing.",
                ),
            ),
            (
                "persona_tools",
                "TOOLS",
                targets["persona_tools"],
                self.tr(
                    "도구 사용 규칙/작업 습관/명령 규약을 입력하세요.",
                    "Write tool usage rules/workflow conventions.",
                ),
            ),
            (
                "persona_heartbeat",
                "HEARTBEAT",
                targets["persona_heartbeat"],
                self.tr(
                    "헬스체크 시 점검할 체크리스트를 입력하세요.",
                    "Write heartbeat checklist items.",
                ),
            ),
        ]
        self.left_tools_panel = self.build_side_panel(self.tr("Persona 파일", "Persona Files"), "left", left_tabs)
        self.left_tools_panel.setVisible(self.left_panel_visible)
        center_layout.addWidget(self.left_tools_panel)

        # Zone B: Chat Area (중단 80%)
        chat_area = QTextBrowser()
        chat_area.setReadOnly(True)
        chat_area.setOpenLinks(False)
        chat_area.anchorClicked.connect(self.on_chat_anchor_clicked)
        chat_area.setPlaceholderText("대화가 여기 표시됩니다...")
        chat_area.setStyleSheet("""
            QTextEdit {
                background: rgba(27, 31, 39, 0.9);
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                font-family: "Segoe UI", "Malgun Gothic", sans-serif;
                line-height: 1.45;
            }
        """)
        center_layout.addWidget(chat_area, 1)
        self.chat_area = chat_area

        self.right_tools_panel = self.build_control_panel(targets)
        self.right_tools_panel.setVisible(self.right_panel_visible)
        center_layout.addWidget(self.right_tools_panel)
        self.update_side_panel_visibility()
        layout.addWidget(center_container, 1)
        
        # Zone C: Input Bar (하단 20%)
        input_container = QWidget()
        input_container.setFixedHeight(120)
        input_container.setStyleSheet("""
            QWidget {
                background: rgba(21, 23, 28, 0.95);
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }
        """)
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(12, 8, 12, 8)
        input_layout.setSpacing(6)
        
        # 입력창
        input_line = QLineEdit()
        input_line.setPlaceholderText("메시지 입력... (Enter로 전송)")
        input_line.setStyleSheet("""
            QLineEdit {
                background: rgba(14, 15, 18, 0.8);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #6bd0ff;
            }
        """)
        input_line.setMinimumHeight(36)
        input_line.returnPressed.connect(self.send_message)
        input_layout.addWidget(input_line)
        self.input_line = input_line
        
        # 전송 버튼
        btn_send = QPushButton("전송")
        btn_send.setStyleSheet("""
            QPushButton {
                background: #6bd0ff;
                color: #071018;
                border: none;
                padding: 6px;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #5bb8e0;
            }
            QPushButton:disabled {
                background: rgba(107, 208, 255, 0.3);
                color: rgba(7, 16, 24, 0.5);
            }
        """)
        btn_send.clicked.connect(self.send_message)
        input_layout.addWidget(btn_send)
        self.btn_send = btn_send

        # Resize grip (bottom-right)
        grip_row = QHBoxLayout()
        grip_row.addStretch()
        size_grip = QSizeGrip(input_container)
        size_grip.setFixedSize(16, 16)
        grip_row.addWidget(size_grip)
        input_layout.addLayout(grip_row)
        
        layout.addWidget(input_container)
        
        # 전체 스타일
        central.setStyleSheet("""
            QWidget {
                border-radius: 12px;
            }
        """)
        
        self.ready = False
        self.debug_enabled = False
        self.processing_timer = QTimer(self)
        self.processing_timer.setInterval(250)
        self.processing_timer.timeout.connect(self.update_processing_indicator)
        self.enable_window_mouse_controls()

    def update_opacity(self, value):
        """투명도 업데이트"""
        self.setWindowOpacity(value / 100.0)

    def split_latency_suffix(self, text: str) -> tuple[str, Optional[float]]:
        raw = (text or "").strip()
        if not raw:
            return "", None
        m = re.search(r"\(응답\s*([0-9]+(?:\.[0-9]+)?)초\)\s*$", raw)
        if not m:
            return raw, None
        try:
            latency = float(m.group(1))
        except Exception:
            latency = None
        return raw[: m.start()].rstrip(), latency

    def extract_structured_payload_text(self, text: str) -> Optional[str]:
        raw = (text or "").strip()
        if not raw or not raw.startswith("{"):
            return None
        obj = None
        try:
            obj = json.loads(raw)
        except Exception:
            try:
                obj = ast.literal_eval(raw)
            except Exception:
                return None
        if not isinstance(obj, dict):
            return None
        result = obj.get("result")
        if not isinstance(result, dict):
            return None
        payloads = result.get("payloads")
        if not isinstance(payloads, list):
            return None
        texts: list[str] = []
        for item in payloads:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item.get("text")))
        joined = "\n".join(part.strip() for part in texts if str(part).strip()).strip()
        return joined or None

    def extract_and_strip_reasoning_blocks(self, text: str) -> tuple[str, list[str]]:
        cleaned = text
        extracted: list[str] = []
        patterns = [
            r"(?is)<think>(.*?)</think>",
            r"(?is)<thinking>(.*?)</thinking>",
            r"(?is)<reasoning>(.*?)</reasoning>",
            r"(?is)<analysis>(.*?)</analysis>",
            r"(?is)```(?:thinking|analysis|reasoning)[^\n]*\n(.*?)```",
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, cleaned))
            for match in matches:
                chunk = (match.group(1) or "").strip()
                if chunk:
                    extracted.append(chunk)
            cleaned = re.sub(pattern, "", cleaned)
        return cleaned.strip(), extracted

    def should_collapse_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*;?", stripped):
            return True
        lowered = stripped.lower()
        collapse_prefixes = (
            "tool:",
            "tool call:",
            "thinking:",
            "reasoning:",
            "analysis:",
        )
        return lowered.startswith(collapse_prefixes)

    def normalize_agent_text(self, raw_message: str) -> tuple[str, Optional[float], str]:
        text, latency = self.split_latency_suffix(raw_message)
        structured = self.extract_structured_payload_text(text)
        if structured:
            text = structured
        text, hidden_blocks = self.extract_and_strip_reasoning_blocks(text)
        hidden_reasoning: list[str] = []
        hidden_reasoning.extend(hidden_blocks)
        filtered_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if not stripped:
                filtered_lines.append(line)
                continue
            if lowered == "an unknown error occurred":
                continue
            if lowered in {"<think>", "</think>", "<thinking>", "</thinking>", "<analysis>", "</analysis>"}:
                hidden_reasoning.append(stripped)
                continue
            if lowered in {"heartbeat", "heartbeat:", "_heartbeat ok!_", "heartbeat_ok", "heartbeat ok"}:
                hidden_reasoning.append(stripped)
                continue
            if stripped in {"지금 번역하기", "번역결과"}:
                hidden_reasoning.append(stripped)
                continue
            if lowered.startswith("#context_"):
                hidden_reasoning.append(stripped)
                continue
            filtered_lines.append(line)
        text = "\n".join(filtered_lines).strip()
        if self.collapse_reasoning_check.isChecked():
            lines: list[str] = []
            for line in text.splitlines():
                if self.should_collapse_line(line):
                    if line.strip():
                        hidden_reasoning.append(line.strip())
                    continue
                lines.append(line)
            compact = "\n".join(lines).strip()
            if compact:
                text = compact
        hidden_text = "\n\n".join(part.strip() for part in hidden_reasoning if part.strip()).strip()
        return text.strip(), latency, hidden_text

    def register_hidden_reasoning(self, reasoning_text: str) -> tuple[Optional[str], int]:
        normalized = (reasoning_text or "").strip()
        if not normalized:
            return None, 0
        self.hidden_reasoning_counter += 1
        reasoning_id = f"reason-{self.hidden_reasoning_counter}"
        self.hidden_reasoning_store[reasoning_id] = normalized
        line_count = len([line for line in normalized.splitlines() if line.strip()])
        return reasoning_id, line_count

    def format_inline_text(self, line: str) -> str:
        escaped = html.escape(line, quote=False)
        escaped = re.sub(
            r"`([^`]+)`",
            lambda m: (
                "<code style='background: rgba(255,255,255,0.08); "
                "padding: 1px 5px; border-radius: 4px;'>"
                f"{m.group(1)}</code>"
            ),
            escaped,
        )
        return escaped

    def text_to_html(self, text: str) -> str:
        lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        parts: list[str] = []
        in_ul = False
        in_ol = False
        blank_streak = 0

        def close_lists():
            nonlocal in_ul, in_ol
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            if in_ol:
                parts.append("</ol>")
                in_ol = False

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                close_lists()
                blank_streak += 1
                if blank_streak <= 1:
                    parts.append("<div style='height:6px'></div>")
                continue
            blank_streak = 0

            m_ul = re.match(r"^[-*•]\s+(.+)$", line)
            m_ol = re.match(r"^(\d+)\.\s+(.+)$", line)
            if m_ul:
                if in_ol:
                    parts.append("</ol>")
                    in_ol = False
                if not in_ul:
                    parts.append("<ul style='margin:4px 0 6px 18px; padding:0;'>")
                    in_ul = True
                parts.append(f"<li style='margin:2px 0'>{self.format_inline_text(m_ul.group(1))}</li>")
                continue
            if m_ol:
                if in_ul:
                    parts.append("</ul>")
                    in_ul = False
                if not in_ol:
                    parts.append("<ol style='margin:4px 0 6px 20px; padding:0;'>")
                    in_ol = True
                parts.append(f"<li style='margin:2px 0'>{self.format_inline_text(m_ol.group(2))}</li>")
                continue

            close_lists()
            parts.append(f"<div style='margin:2px 0'>{self.format_inline_text(line)}</div>")

        close_lists()
        html_body = "".join(parts).strip()
        if not html_body:
            return "<div style='color:#9aa0a6'>(빈 응답)</div>"
        return html_body

    def render_assistant_block(
        self,
        message: str,
        latency: Optional[float] = None,
        hidden_reasoning_id: Optional[str] = None,
        hidden_reasoning_lines: int = 0,
    ) -> str:
        body = self.text_to_html(message)
        latency_html = ""
        if latency is not None:
            latency_html = (
                f"<div style='margin-top:5px; color:#8d97a6; font-size:11px;'>"
                f"응답 {latency:.1f}초</div>"
            )
        folded_html = ""
        if hidden_reasoning_id:
            folded_html = (
                "<div style='margin-top:6px; font-size:11px;'>"
                f"<a href='reason://{hidden_reasoning_id}' "
                "style='color:#8fd8ff; text-decoration:none;'>"
                f"▶ {self.tr('접힌 중간 과정 보기', 'Show folded reasoning')} ({hidden_reasoning_lines})"
                "</a></div>"
            )
        return (
            "<div style='margin: 8px 0;'>"
            "<div style='color:#8fd8ff; font-weight:600; margin-bottom:2px;'>👻</div>"
            f"<div style='padding:8px 10px; background: rgba(255,255,255,0.04); "
            f"border:1px solid rgba(255,255,255,0.08); border-radius:10px;'>{body}{latency_html}{folded_html}</div>"
            "</div>"
        )

    def append_debug(self, message: str):
        if not self.debug_toggle.isChecked():
            return
        self.chat_area.append(f"<span style='color:#9aa0a6'>{message}</span>")
        self.chat_area.moveCursor(QTextCursor.End)

    def append_system_line(self, text: str):
        self.chat_area.append(text)
        self.chat_area.moveCursor(QTextCursor.End)

    def format_short_number(self, value: int) -> str:
        try:
            numeric = int(value)
        except Exception:
            return str(value)
        if numeric >= 1_000_000:
            return f"{numeric / 1_000_000:.1f}M"
        if numeric >= 1000:
            return f"{numeric / 1000:.1f}k"
        return str(numeric)

    def format_runtime_stats_text(self, stats: dict[str, Any]) -> tuple[str, str]:
        context_used = clamp_int(stats.get("contextUsed"), 0, 0, 10_000_000)
        context_window = clamp_int(stats.get("contextWindow"), 0, 0, 10_000_000)
        context_remaining = clamp_int(stats.get("contextRemaining"), 0, 0, 10_000_000)
        usage_pct = float(stats.get("contextUsagePct") or 0.0)
        tok_per_sec = float(stats.get("tokensPerSec") or 0.0)
        latency_sec = float(stats.get("latencySec") or 0.0)
        prompt_tokens = clamp_int(stats.get("promptTokens"), 0, 0, 10_000_000)
        completion_tokens = clamp_int(stats.get("completionTokens"), 0, 0, 10_000_000)
        model_name = str(stats.get("model") or "").strip()
        kv_used_mb = stats.get("kvUsedMB")
        kv_remaining_mb = stats.get("kvRemainingMB")
        kv_total_mb = stats.get("kvTotalMB")

        if context_window > 0:
            short_ctx = (
                f"Ctx {self.format_short_number(context_used)}/{self.format_short_number(context_window)} "
                f"({usage_pct:.1f}%)"
            )
        else:
            short_ctx = f"Ctx {self.format_short_number(context_used)}"

        if isinstance(kv_remaining_mb, (int, float)):
            short_kv = f"KV 남음 {kv_remaining_mb/1024.0:.1f}G" if kv_remaining_mb >= 1024 else f"KV 남음 {kv_remaining_mb:.0f}M"
        else:
            short_kv = "KV 추정 N/A"
        short_speed = f"{tok_per_sec:.1f} tok/s" if tok_per_sec > 0 else "tok/s -"
        short_text = f"{short_ctx} | {short_kv} | {short_speed}"

        detail_lines = [
            f"model: {model_name or 'unknown'}",
            f"context used: {context_used:,} / {context_window:,} (remaining {context_remaining:,}, {usage_pct:.2f}%)",
            f"prompt tokens: {prompt_tokens:,}",
            f"completion tokens: {completion_tokens:,}",
            f"latency: {latency_sec:.2f}s",
            f"tokens/sec: {tok_per_sec:.2f}",
        ]
        if isinstance(kv_used_mb, (int, float)) and isinstance(kv_total_mb, (int, float)) and isinstance(kv_remaining_mb, (int, float)):
            detail_lines.append(
                f"estimated KV cache: used {kv_used_mb:.1f} MB / total {kv_total_mb:.1f} MB (remaining {kv_remaining_mb:.1f} MB)"
            )
        else:
            detail_lines.append("estimated KV cache: not available for this model metadata")
        return short_text, "\n".join(detail_lines)

    def on_agent_stats(self, stats_obj: object):
        if not isinstance(stats_obj, dict):
            return
        self.last_runtime_stats = dict(stats_obj)
        short_text, detail_text = self.format_runtime_stats_text(self.last_runtime_stats)
        if hasattr(self, "runtime_stats_label"):
            self.runtime_stats_label.setText(short_text)
            self.runtime_stats_label.setToolTip(detail_text)

    def refresh_settings_cache(self) -> dict:
        settings = load_settings()
        self.settings_cache = settings if isinstance(settings, dict) else {}
        profiles = set(list_profiles(self.settings_cache))
        if normalize_profile_id(self.profile_id) not in profiles:
            self.profile_id = get_active_profile_id(self.settings_cache)
        self.profile_settings = get_profile(self.settings_cache, self.profile_id)
        self.ui_language = self.resolve_ui_language(self.settings_cache)
        return self.settings_cache

    def merge_and_save_settings(self, patch_data: dict):
        settings = self.refresh_settings_cache()
        profiles = settings.get("profiles", {}) if isinstance(settings.get("profiles"), dict) else {}
        profile_data = get_profile(settings, self.profile_id)
        global_keys = {"alwaysOnTop", "ui", "activeProfile", "profiles"}
        for key, value in patch_data.items():
            if key in global_keys:
                settings[key] = value
            else:
                profile_data[key] = value
        profiles[self.profile_id] = profile_data
        settings["profiles"] = profiles
        save_settings(settings)
        self.refresh_settings_cache()

    def extract_json_payload(self, output: str) -> Optional[Any]:
        text = (output or "").strip()
        if not text:
            return None
        for candidate in [text]:
            try:
                return json.loads(candidate)
            except Exception:
                pass
        for opener, closer in [("{", "}"), ("[", "]")]:
            start = text.find(opener)
            end = text.rfind(closer)
            if start == -1 or end == -1 or end <= start:
                continue
            chunk = text[start : end + 1]
            try:
                return json.loads(chunk)
            except Exception:
                continue
        return None

    def parse_cli_value(self, output: str) -> Any:
        parsed = self.extract_json_payload(output)
        if parsed is not None:
            return parsed
        text = (output or "").strip()
        lower = text.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower in {"null", "none"}:
            return None
        return text

    def persist_always_on_top(self, enabled: bool):
        try:
            settings = load_settings()
            settings["alwaysOnTop"] = bool(enabled)
            save_settings(settings)
        except Exception as e:
            logger.warning(f"alwaysOnTop 저장 실패: {e}")

    def sync_always_on_top_controls(self):
        if hasattr(self, "always_on_top_toggle"):
            self.always_on_top_toggle.blockSignals(True)
            self.always_on_top_toggle.setChecked(self.always_on_top)
            self.always_on_top_toggle.blockSignals(False)
        if hasattr(self, "always_on_top_action"):
            self.always_on_top_action.blockSignals(True)
            self.always_on_top_action.setChecked(self.always_on_top)
            self.always_on_top_action.blockSignals(False)

    def apply_always_on_top(self, enabled: bool, persist: bool = True):
        enabled = bool(enabled)
        self.always_on_top = enabled
        was_visible = self.isVisible()
        self.setWindowFlag(Qt.WindowStaysOnTopHint, enabled)
        if was_visible:
            self.show()
        self.sync_always_on_top_controls()
        if persist:
            self.persist_always_on_top(enabled)

    def on_always_on_top_toggled(self, checked: bool):
        self.apply_always_on_top(checked, persist=True)

    def normalize_watch_path(self, path_value: Path | str) -> str:
        try:
            return str(Path(path_value).resolve()).lower()
        except Exception:
            return str(path_value).lower()

    def file_signature(self, path: Path) -> Optional[tuple[int, int]]:
        try:
            if not path.exists():
                return None
            stat = path.stat()
            return (int(stat.st_mtime_ns), int(stat.st_size))
        except Exception:
            return None

    def refresh_editor_watch_targets(self):
        desired: set[str] = set()
        for file_path in self.editor_file_paths.values():
            desired.add(str(file_path))
            desired.add(str(file_path.parent))

        desired_keys = {self.normalize_watch_path(item) for item in desired}

        current_paths = list(self.file_watcher.files()) + list(self.file_watcher.directories())
        stale_paths = [path for path in current_paths if self.normalize_watch_path(path) not in desired_keys]
        if stale_paths:
            self.file_watcher.removePaths(stale_paths)

        existing = {self.normalize_watch_path(path) for path in list(self.file_watcher.files()) + list(self.file_watcher.directories())}
        to_add = [path for path in desired if self.normalize_watch_path(path) not in existing]
        if to_add:
            self.file_watcher.addPaths(to_add)
        self.file_watch_paths = desired_keys

    def editor_ids_for_changed_path(self, changed_path: str) -> list[str]:
        changed_key = self.normalize_watch_path(changed_path)
        matches: list[str] = []
        for editor_id, file_path in self.editor_file_paths.items():
            file_key = self.normalize_watch_path(file_path)
            dir_key = self.normalize_watch_path(file_path.parent)
            if changed_key in {file_key, dir_key}:
                matches.append(editor_id)
        return matches

    def queue_external_refresh_for_path(self, changed_path: str):
        matched_ids = self.editor_ids_for_changed_path(changed_path)
        if not matched_ids:
            return
        for editor_id in matched_ids:
            self.pending_external_refresh_ids.add(editor_id)
        if not self.external_refresh_timer.isActive():
            self.external_refresh_timer.start()

    def on_watched_file_changed(self, changed_path: str):
        self.queue_external_refresh_for_path(changed_path)
        self.refresh_editor_watch_targets()

    def on_watched_directory_changed(self, changed_path: str):
        self.queue_external_refresh_for_path(changed_path)
        self.refresh_editor_watch_targets()

    def flush_pending_external_refresh(self):
        if not self.pending_external_refresh_ids:
            return
        editor_ids = sorted(self.pending_external_refresh_ids)
        self.pending_external_refresh_ids.clear()
        refreshed = 0
        for editor_id in editor_ids:
            if self.load_editor_content(editor_id, source="watch", force=False):
                refreshed += 1
        if refreshed > 0:
            self.status_label.setText(
                self.tr(
                    f"🔄 외부 파일 변경 감지: {refreshed}개 자동 새로고침",
                    f"🔄 External file change detected: auto-refreshed {refreshed}",
                )
            )

    def get_editor_targets(self) -> dict[str, Path]:
        workspace_dir = self.get_active_workspace_dir()
        daily_memory_dir = workspace_dir / "memory"
        daily_memory_file = daily_memory_dir / f"{date.today().isoformat()}.md"
        return {
            "persona_agents": workspace_dir / "AGENTS.md",
            "persona_soul": workspace_dir / "SOUL.md",
            "persona_identity": workspace_dir / "IDENTITY.md",
            "persona_user": workspace_dir / "USER.md",
            "persona_tools": workspace_dir / "TOOLS.md",
            "persona_heartbeat": workspace_dir / "HEARTBEAT.md",
            "memory_daily": daily_memory_file,
            "memory_long": workspace_dir / "MEMORY.md",
        }

    def get_active_agent_id(self) -> str:
        profile = self.profile_settings if isinstance(self.profile_settings, dict) else {}
        return normalize_agent_id(str(profile.get("agentId") or self.profile_id or "main"))

    def get_active_workspace_dir(self) -> Path:
        profile = self.profile_settings if isinstance(self.profile_settings, dict) else {}
        return resolve_workspace_path(self.gateway_manager.repo_root, profile.get("workspace"))

    def load_editor_content(self, editor_id: str, source: str = "manual", force: bool = True) -> bool:
        text_edit = self.editor_widgets.get(editor_id)
        file_path = self.editor_file_paths.get(editor_id)
        if not text_edit or not file_path:
            return False
        latest_signature = self.file_signature(file_path)
        if not force:
            previous_signature = self.editor_file_signatures.get(editor_id)
            if latest_signature == previous_signature:
                return False
            if text_edit.document().isModified():
                self.append_debug(
                    f"외부 변경 감지했지만 미저장 편집이 있어 자동 새로고침 보류: {file_path}"
                )
                self.status_label.setText(
                    self.tr(
                        "⚠️ 외부 변경 감지됨 (미저장 편집 때문에 자동 새로고침 보류)",
                        "⚠️ External change detected (auto refresh paused due to unsaved edits)",
                    )
                )
                return False
        try:
            content = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
        except Exception as e:
            content = ""
            self.append_debug(f"파일 읽기 실패: {file_path} ({e})")
        cursor = text_edit.textCursor()
        cursor_pos = cursor.position()
        text_edit.setPlainText(content)
        if content:
            cursor = text_edit.textCursor()
            cursor.setPosition(min(cursor_pos, len(content)))
            text_edit.setTextCursor(cursor)
        text_edit.document().setModified(False)
        self.editor_file_signatures[editor_id] = latest_signature
        if source == "watch":
            self.append_debug(f"자동 새로고침: {file_path}")
        return True

    def save_editor_content(self, editor_id: str):
        text_edit = self.editor_widgets.get(editor_id)
        file_path = self.editor_file_paths.get(editor_id)
        if not text_edit or not file_path:
            return
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(text_edit.toPlainText(), encoding="utf-8")
            text_edit.document().setModified(False)
            self.editor_file_signatures[editor_id] = self.file_signature(file_path)
            self.refresh_editor_watch_targets()
            self.status_label.setText("✅ 메모리/페르소나 저장됨")
            self.append_debug(f"저장 완료: {file_path}")
        except Exception as e:
            self.status_label.setText("❌ 저장 실패")
            self.append_debug(f"저장 실패: {file_path} ({e})")

    def create_editor_tab(
        self,
        editor_id: str,
        title: str,
        file_path: Path,
        placeholder: str,
    ) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        path_label = QLabel(str(file_path))
        path_label.setStyleSheet("color: rgba(255,255,255,0.55); font-size: 10px;")
        path_label.setWordWrap(True)
        layout.addWidget(path_label)

        editor = QTextEdit()
        editor.setAcceptRichText(False)
        editor.setPlaceholderText(placeholder)
        editor.setStyleSheet("""
            QTextEdit {
                background: rgba(13, 15, 20, 0.92);
                color: #e9eef7;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
                font-family: "Consolas", "D2Coding", "Malgun Gothic", monospace;
                line-height: 1.45;
            }
        """)
        layout.addWidget(editor, 1)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)
        reload_btn = QPushButton("불러오기")
        reload_btn.setFixedHeight(26)
        save_btn = QPushButton("저장")
        save_btn.setFixedHeight(26)
        reload_btn.clicked.connect(lambda _=False, eid=editor_id: self.load_editor_content(eid))
        save_btn.clicked.connect(lambda _=False, eid=editor_id: self.save_editor_content(eid))
        button_row.addWidget(reload_btn)
        button_row.addWidget(save_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.editor_file_paths[editor_id] = file_path
        self.editor_path_labels[editor_id] = path_label
        self.editor_widgets[editor_id] = editor
        self.load_editor_content(editor_id)
        return container

    def update_editor_targets_for_profile(self):
        targets = self.get_editor_targets()
        mapping = {
            "persona_agents": targets["persona_agents"],
            "persona_soul": targets["persona_soul"],
            "persona_identity": targets["persona_identity"],
            "persona_user": targets["persona_user"],
            "persona_tools": targets["persona_tools"],
            "persona_heartbeat": targets["persona_heartbeat"],
            "memory_daily": targets["memory_daily"],
            "memory_long": targets["memory_long"],
        }
        for editor_id, target_path in mapping.items():
            if editor_id not in self.editor_file_paths:
                continue
            self.editor_file_paths[editor_id] = target_path
            label = self.editor_path_labels.get(editor_id)
            if label:
                label.setText(str(target_path))
            self.load_editor_content(editor_id)
        self.refresh_editor_watch_targets()

    def build_side_panel(self, panel_title: str, side: str, tabs: list[tuple[str, str, Path, str]]) -> QFrame:
        panel = QFrame()
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(360)
        border_side = "right" if side == "left" else "left"
        panel.setStyleSheet(f"""
            QFrame {{
                background: rgba(18, 21, 27, 0.96);
                border-{border_side}: 1px solid rgba(255,255,255,0.11);
            }}
            QLabel {{
                color: #e5e7eb;
            }}
            QTabWidget::pane {{
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 8px;
                background: rgba(20, 23, 31, 0.9);
            }}
            QTabBar::tab {{
                background: rgba(255,255,255,0.06);
                color: #cfd7e3;
                padding: 6px 10px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 3px;
            }}
            QTabBar::tab:selected {{
                background: rgba(107, 208, 255, 0.2);
                color: #ffffff;
            }}
            QPushButton {{
                background: rgba(255,255,255,0.10);
                color: #f3f4f6;
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 6px;
                padding: 4px 10px;
            }}
            QPushButton:hover {{
                background: rgba(255,255,255,0.18);
            }}
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title_label = QLabel(panel_title)
        title_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #dbe7ff;")
        layout.addWidget(title_label)

        tab_widget = QTabWidget()
        for editor_id, label, file_path, placeholder in tabs:
            tab_widget.addTab(self.create_editor_tab(editor_id, label, file_path, placeholder), label)
        layout.addWidget(tab_widget, 1)
        return panel

    def build_control_panel(self, targets: dict[str, Path]) -> QFrame:
        panel = QFrame()
        panel.setMinimumWidth(360)
        panel.setMaximumWidth(460)
        panel.setStyleSheet("""
            QFrame {
                background: rgba(18, 21, 27, 0.96);
                border-left: 1px solid rgba(255,255,255,0.11);
            }
            QLabel {
                color: #e5e7eb;
            }
            QTabWidget::pane {
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 8px;
                background: rgba(20, 23, 31, 0.9);
            }
            QTabBar::tab {
                background: rgba(255,255,255,0.06);
                color: #cfd7e3;
                padding: 6px 10px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 3px;
            }
            QTabBar::tab:selected {
                background: rgba(107, 208, 255, 0.2);
                color: #ffffff;
            }
            QLineEdit, QTextEdit, QComboBox {
                background: rgba(13, 15, 20, 0.92);
                color: #e9eef7;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 8px;
                padding: 6px;
            }
            QPushButton {
                background: rgba(255,255,255,0.10);
                color: #f3f4f6;
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 6px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background: rgba(255,255,255,0.18);
            }
            QCheckBox {
                color: #d1d5db;
                font-size: 11px;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title_label = QLabel(self.tr("메모리 / 세션 / 스킬", "Memory / Session / Skills"))
        title_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #dbe7ff;")
        layout.addWidget(title_label)

        tab_widget = QTabWidget()
        tab_widget.addTab(self.create_memory_tab(targets), self.tr("메모리", "Memory"))
        tab_widget.addTab(self.create_session_tab(), self.tr("세션", "Session"))
        tab_widget.addTab(self.create_skills_tab(), self.tr("스킬", "Skills"))
        layout.addWidget(tab_widget, 1)
        return panel

    def create_memory_tab(self, targets: dict[str, Path]) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.memory_enabled_check = QCheckBox(
            self.tr(
                "메모리 검색 사용 (memorySearch.enabled)",
                "Enable memory search (memorySearch.enabled)",
            )
        )
        self.memory_enabled_check.setChecked(True)
        self.memory_source_sessions_check = QCheckBox(
            self.tr(
                "세션 기록도 검색 (sources includes sessions)",
                "Include sessions in sources",
            )
        )
        self.memory_session_experimental_check = QCheckBox(
            self.tr(
                "세션 메모리 실험 기능 (experimental.sessionMemory)",
                "Experimental session memory (experimental.sessionMemory)",
            )
        )
        self.memory_sync_start_check = QCheckBox(
            self.tr(
                "세션 시작 시 메모리 동기화 (sync.onSessionStart)",
                "Sync on session start (sync.onSessionStart)",
            )
        )
        self.memory_sync_start_check.setChecked(True)
        layout.addWidget(self.memory_enabled_check)
        layout.addWidget(self.memory_source_sessions_check)
        layout.addWidget(self.memory_session_experimental_check)
        layout.addWidget(self.memory_sync_start_check)

        control_row = QHBoxLayout()
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(6)
        reload_btn = QPushButton(self.tr("설정 불러오기", "Load Config"))
        apply_btn = QPushButton(self.tr("설정 저장", "Save Config"))
        reload_btn.clicked.connect(self.refresh_memory_controls_from_config)
        apply_btn.clicked.connect(self.save_memory_controls_to_config)
        control_row.addWidget(reload_btn)
        control_row.addWidget(apply_btn)
        control_row.addStretch()
        layout.addLayout(control_row)

        memory_note = QLabel(
            self.tr(
                "안내: 파일 저장은 즉시 디스크 반영됩니다. config 변경은 게이트웨이 재시작 후 안정적으로 반영됩니다.",
                "Note: File saves are immediate. Config changes are applied reliably after gateway restart.",
            )
        )
        memory_note.setWordWrap(True)
        memory_note.setStyleSheet("color: rgba(255,255,255,0.58); font-size: 10px;")
        layout.addWidget(memory_note)

        files_tabs = QTabWidget()
        files_tabs.addTab(
            self.create_editor_tab(
                "memory_daily",
                "Today",
                targets["memory_daily"],
                self.tr(
                    "오늘 메모를 기록하세요. (workspace/memory/YYYY-MM-DD.md)",
                    "Write daily notes here. (workspace/memory/YYYY-MM-DD.md)",
                ),
            ),
            self.tr("일일 메모", "Daily"),
        )
        files_tabs.addTab(
            self.create_editor_tab(
                "memory_long",
                "MEMORY",
                targets["memory_long"],
                self.tr(
                    "장기 기억(MEMORY.md)을 정리하세요.",
                    "Organize long-term memory (MEMORY.md).",
                ),
            ),
            "MEMORY.md",
        )
        layout.addWidget(files_tabs, 1)
        return container

    def get_agent_list_entries(self) -> list[dict]:
        ok, out = self.run_openclaw_command(["config", "get", "--json", "agents.list"])
        if not ok:
            return []
        parsed = self.parse_cli_value(out)
        if isinstance(parsed, list):
            return [entry for entry in parsed if isinstance(entry, dict)]
        return []

    def find_agent_index(self, agent_id: str) -> Optional[int]:
        target = normalize_agent_id(agent_id)
        for idx, entry in enumerate(self.get_agent_list_entries()):
            if normalize_agent_id(str(entry.get("id") or "")) == target:
                return idx
        return None

    def resolve_memory_search_base_path(self) -> str:
        agent_id = self.get_active_agent_id()
        index = self.find_agent_index(agent_id)
        if index is None:
            return "agents.defaults.memorySearch"
        return f"agents.list[{index}].memorySearch"

    def refresh_memory_controls_from_config(self):
        defaults = {
            "enabled": True,
            "sources": ["memory"],
            "session_memory": False,
            "on_session_start": True,
        }
        base_path = self.resolve_memory_search_base_path()
        try:
            ok, out = self.run_openclaw_command(["config", "get", f"{base_path}.enabled"])
            if ok:
                parsed = self.parse_cli_value(out)
                if isinstance(parsed, bool):
                    defaults["enabled"] = parsed

            ok, out = self.run_openclaw_command(["config", "get", f"{base_path}.sources"])
            if ok:
                parsed = self.parse_cli_value(out)
                if isinstance(parsed, list):
                    defaults["sources"] = [str(item) for item in parsed]

            ok, out = self.run_openclaw_command(
                ["config", "get", f"{base_path}.experimental.sessionMemory"]
            )
            if ok:
                parsed = self.parse_cli_value(out)
                if isinstance(parsed, bool):
                    defaults["session_memory"] = parsed

            ok, out = self.run_openclaw_command(
                ["config", "get", f"{base_path}.sync.onSessionStart"]
            )
            if ok:
                parsed = self.parse_cli_value(out)
                if isinstance(parsed, bool):
                    defaults["on_session_start"] = parsed
        except Exception as e:
            self.append_debug(f"메모리 설정 조회 실패: {e}")

        self.memory_enabled_check.setChecked(bool(defaults["enabled"]))
        self.memory_source_sessions_check.setChecked("sessions" in defaults["sources"])
        self.memory_session_experimental_check.setChecked(bool(defaults["session_memory"]))
        self.memory_sync_start_check.setChecked(bool(defaults["on_session_start"]))
        self.status_label.setText(self.tr("✅ 메모리 설정 불러오기 완료", "✅ Memory config loaded"))

    def save_memory_controls_to_config(self):
        enabled = self.memory_enabled_check.isChecked()
        include_sessions = self.memory_source_sessions_check.isChecked()
        session_memory = self.memory_session_experimental_check.isChecked()
        on_session_start = self.memory_sync_start_check.isChecked()
        sources = ["memory"] + (["sessions"] if include_sessions else [])
        base_path = self.resolve_memory_search_base_path()

        commands = [
            ["config", "set", "--json", f"{base_path}.enabled", json.dumps(enabled)],
            ["config", "set", "--json", f"{base_path}.sources", json.dumps(sources)],
            [
                "config",
                "set",
                "--json",
                f"{base_path}.experimental.sessionMemory",
                json.dumps(session_memory),
            ],
            [
                "config",
                "set",
                "--json",
                f"{base_path}.sync.onSessionStart",
                json.dumps(on_session_start),
            ],
        ]
        failed: list[str] = []
        for args in commands:
            ok, out = self.run_openclaw_command(args)
            if not ok:
                failed.append(f"{' '.join(args)} => {out}")
        if failed:
            self.status_label.setText(self.tr("❌ 메모리 설정 저장 실패", "❌ Failed to save memory config"))
            self.append_system_line(
                "<b>"
                + self.tr("❌ 메모리 설정 저장 실패", "❌ Failed to save memory config")
                + "</b><br>"
                + "<br>".join(failed)
            )
            return
        self.status_label.setText(
            self.tr(
                "✅ 메모리 설정 저장 완료 (게이트웨이 재시작 권장)",
                "✅ Memory config saved (gateway restart recommended)",
            )
        )

    def create_session_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        session_key_row = QHBoxLayout()
        session_key_row.setContentsMargins(0, 0, 0, 0)
        session_key_row.addWidget(QLabel(self.tr("GhostChat 세션 키", "GhostChat Session Key")))
        self.session_key_input = QLineEdit(str(self.profile_settings.get("sessionKey") or "").strip())
        self.session_key_input.setPlaceholderText(
            self.tr("비워두면 런타임 키 사용", "Leave blank to use runtime key")
        )
        session_key_row.addWidget(self.session_key_input, 1)
        layout.addLayout(session_key_row)

        dm_scope_row = QHBoxLayout()
        dm_scope_row.setContentsMargins(0, 0, 0, 0)
        dm_scope_row.addWidget(
            QLabel(self.tr("DM 세션 범위 (session.dmScope)", "DM Scope (session.dmScope)"))
        )
        self.dm_scope_combo = QComboBox()
        self.dm_scope_combo.addItems(
            ["main", "per-peer", "per-channel-peer", "per-account-channel-peer"]
        )
        self.dm_scope_combo.setCurrentText("main")
        dm_scope_row.addWidget(self.dm_scope_combo, 1)
        layout.addLayout(dm_scope_row)

        inject_note = QLabel(
            self.tr(
                "주입 동작: 새 sessionKey의 첫 메시지에서 AGENTS/SOUL/TOOLS/IDENTITY/USER 등이 자동 주입됩니다.",
                "Injection behavior: On the first message of a new sessionKey, AGENTS/SOUL/TOOLS/IDENTITY/USER are auto-injected.",
            )
        )
        inject_note.setWordWrap(True)
        inject_note.setStyleSheet("color: rgba(255,255,255,0.58); font-size: 10px;")
        layout.addWidget(inject_note)

        layout.addWidget(
            QLabel(self.tr("사용자 연결 규칙 (session.identityLinks)", "Identity Links (session.identityLinks)"))
        )
        self.identity_links_table = QTableWidget(0, 2)
        self.identity_links_table.setHorizontalHeaderLabels(
            [
                self.tr("대표 ID", "Canonical ID"),
                self.tr("연결 ID들(쉼표 구분)", "Linked IDs (comma separated)"),
            ]
        )
        self.identity_links_table.setMinimumHeight(150)
        self.identity_links_table.setColumnWidth(0, 150)
        self.identity_links_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.identity_links_table, 1)

        links_row = QHBoxLayout()
        links_row.setContentsMargins(0, 0, 0, 0)
        links_row.setSpacing(6)
        add_link_btn = QPushButton(self.tr("행 추가", "Add Row"))
        remove_link_btn = QPushButton(self.tr("선택 행 삭제", "Delete Selected"))
        add_link_btn.clicked.connect(lambda: self.add_identity_link_row())
        remove_link_btn.clicked.connect(self.remove_selected_identity_link_rows)
        links_row.addWidget(add_link_btn)
        links_row.addWidget(remove_link_btn)
        links_row.addStretch()
        layout.addLayout(links_row)

        note = QLabel(
            self.tr(
                "예) 대표ID: person:me / 연결ID: telegram:123, discord:abc",
                "Example) canonical: person:me / linked: telegram:123, discord:abc",
            )
        )
        note.setStyleSheet("color: rgba(255,255,255,0.55); font-size: 10px;")
        layout.addWidget(note)

        session_note = QLabel(
            self.tr(
                "안내: Session 설정 저장 후 게이트웨이 재시작 시 안정적으로 반영됩니다.",
                "Note: Session config is applied reliably after gateway restart.",
            )
        )
        session_note.setWordWrap(True)
        session_note.setStyleSheet("color: rgba(255,255,255,0.58); font-size: 10px;")
        layout.addWidget(session_note)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)
        reload_btn = QPushButton(self.tr("설정 불러오기", "Load Config"))
        save_btn = QPushButton(self.tr("설정 저장", "Save Config"))
        reload_btn.clicked.connect(self.refresh_session_controls_from_config)
        save_btn.clicked.connect(self.save_session_controls_to_config)
        button_row.addWidget(reload_btn)
        button_row.addWidget(save_btn)
        button_row.addStretch()
        layout.addLayout(button_row)
        return container

    def add_identity_link_row(self, canonical: str = "", linked: str = ""):
        if not hasattr(self, "identity_links_table"):
            return
        table = self.identity_links_table
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(canonical))
        table.setItem(row, 1, QTableWidgetItem(linked))

    def remove_selected_identity_link_rows(self):
        if not hasattr(self, "identity_links_table"):
            return
        table = self.identity_links_table
        selected_rows = sorted({item.row() for item in table.selectedItems()}, reverse=True)
        if not selected_rows and table.currentRow() >= 0:
            selected_rows = [table.currentRow()]
        for row in selected_rows:
            table.removeRow(row)

    def fill_identity_links_table(self, identity_links: dict[str, Any]):
        if not hasattr(self, "identity_links_table"):
            return
        table = self.identity_links_table
        table.setRowCount(0)
        for canonical in sorted(identity_links.keys()):
            raw = identity_links.get(canonical)
            linked_ids: list[str] = []
            if isinstance(raw, list):
                linked_ids = [str(item).strip() for item in raw if str(item).strip()]
            elif isinstance(raw, str):
                linked_ids = [token.strip() for token in re.split(r"[,\n]+", raw) if token.strip()]
            if not linked_ids:
                continue
            self.add_identity_link_row(str(canonical).strip(), ", ".join(linked_ids))
        if table.rowCount() == 0:
            self.add_identity_link_row()

    def collect_identity_links_from_table(self) -> dict[str, list[str]]:
        identity_links: dict[str, list[str]] = {}
        if not hasattr(self, "identity_links_table"):
            return identity_links
        table = self.identity_links_table
        for row in range(table.rowCount()):
            canonical_item = table.item(row, 0)
            linked_item = table.item(row, 1)
            canonical = canonical_item.text().strip() if canonical_item else ""
            linked_raw = linked_item.text().strip() if linked_item else ""
            if not canonical or not linked_raw:
                continue
            linked_ids = [token.strip() for token in re.split(r"[,\n]+", linked_raw) if token.strip()]
            if not linked_ids:
                continue
            identity_links[canonical] = linked_ids
        return identity_links

    def refresh_session_controls_from_config(self):
        self.refresh_settings_cache()
        self.session_key_input.setText(str(self.profile_settings.get("sessionKey") or "").strip())

        scope = "main"
        identity_links: Any = {}

        ok, out = self.run_openclaw_command(["config", "get", "session.dmScope"])
        if ok:
            parsed = self.parse_cli_value(out)
            if isinstance(parsed, str) and parsed.strip():
                scope = parsed.strip()

        ok, out = self.run_openclaw_command(["config", "get", "session.identityLinks"])
        if ok:
            parsed = self.parse_cli_value(out)
            if isinstance(parsed, dict):
                identity_links = parsed

        idx = self.dm_scope_combo.findText(scope)
        self.dm_scope_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.fill_identity_links_table(identity_links)
        self.status_label.setText(self.tr("✅ 세션 설정 불러오기 완료", "✅ Session config loaded"))

    def save_session_controls_to_config(self):
        session_key = self.session_key_input.text().strip()
        dm_scope = self.dm_scope_combo.currentText().strip() or "main"
        identity_links = self.collect_identity_links_from_table()

        failed: list[str] = []
        self.merge_and_save_settings({"sessionKey": session_key})

        ok, out = self.run_openclaw_command(["config", "set", "session.dmScope", dm_scope])
        if not ok:
            failed.append(f"session.dmScope => {out}")

        ok, out = self.run_openclaw_command(
            ["config", "set", "--json", "session.identityLinks", json.dumps(identity_links, ensure_ascii=False)]
        )
        if not ok:
            failed.append(f"session.identityLinks => {out}")

        if failed:
            self.status_label.setText(self.tr("❌ 세션 설정 저장 실패", "❌ Failed to save session config"))
            self.append_system_line(
                "<b>"
                + self.tr("❌ 세션 설정 저장 실패", "❌ Failed to save session config")
                + "</b><br>"
                + "<br>".join(failed)
            )
            return
        self.status_label.setText(
            self.tr(
                "✅ 세션 설정 저장 완료 (게이트웨이 재시작 권장)",
                "✅ Session config saved (gateway restart recommended)",
            )
        )

    def create_skills_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        row_name = QHBoxLayout()
        row_name.setContentsMargins(0, 0, 0, 0)
        row_name.addWidget(QLabel(self.tr("스킬 이름", "Skill Name")))
        self.skill_name_input = QLineEdit()
        self.skill_name_input.setPlaceholderText(self.tr("예: healthcheck", "e.g. healthcheck"))
        row_name.addWidget(self.skill_name_input, 1)
        enable_btn = QPushButton(self.tr("활성화", "Enable"))
        disable_btn = QPushButton(self.tr("비활성화", "Disable"))
        enable_btn.clicked.connect(lambda: self.set_skill_enabled(True))
        disable_btn.clicked.connect(lambda: self.set_skill_enabled(False))
        row_name.addWidget(enable_btn)
        row_name.addWidget(disable_btn)
        layout.addLayout(row_name)

        row_install = QHBoxLayout()
        row_install.setContentsMargins(0, 0, 0, 0)
        row_install.addWidget(QLabel("ClawHub slug"))
        self.skill_slug_input = QLineEdit()
        self.skill_slug_input.setPlaceholderText(self.tr("예: owner/skill", "e.g. owner/skill"))
        row_install.addWidget(self.skill_slug_input, 1)
        install_btn = QPushButton(self.tr("설치", "Install"))
        install_btn.clicked.connect(self.install_skill_from_slug)
        row_install.addWidget(install_btn)
        layout.addLayout(row_install)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        refresh_btn = QPushButton(self.tr("목록 새로고침", "Refresh List"))
        refresh_btn.clicked.connect(self.refresh_skills_status)
        action_row.addWidget(refresh_btn)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.skills_status_view = QTextEdit()
        self.skills_status_view.setReadOnly(True)
        self.skills_status_view.setMinimumHeight(220)
        self.skills_status_view.setStyleSheet(
            "font-family: 'Consolas', 'D2Coding', 'Malgun Gothic', monospace; font-size: 11px;"
        )
        self.skills_status_view.setPlainText(
            self.tr(
                "목록을 보려면 '목록 새로고침'을 누르세요.",
                "Press 'Refresh List' to load skills.",
            )
        )
        layout.addWidget(self.skills_status_view, 1)
        return container

    def refresh_skills_status(self):
        ok_list, out_list = self.run_openclaw_command(["skills", "list"])
        ok_entries, out_entries = self.run_openclaw_command(["config", "get", "skills.entries"])
        summary_lines = [
            "[skills list]",
            out_list if ok_list else f"실패: {out_list}",
            "",
            "[skills.entries]",
            out_entries if ok_entries else f"실패: {out_entries}",
        ]
        self.skills_status_view.setPlainText("\n".join(summary_lines))
        self.status_label.setText(self.tr("✅ 스킬 상태 갱신 완료", "✅ Skills status refreshed"))

    def set_skill_enabled(self, enabled: bool):
        name = self.skill_name_input.text().strip()
        if not name:
            self.status_label.setText(self.tr("❌ 스킬 이름을 입력하세요", "❌ Enter a skill name"))
            return
        ok, out = self.run_openclaw_command(
            ["config", "set", "--json", f"skills.entries.{name}.enabled", json.dumps(bool(enabled))]
        )
        if ok:
            state = self.tr("활성화", "enabled") if enabled else self.tr("비활성화", "disabled")
            self.status_label.setText(self.tr(f"✅ 스킬 {state}: {name}", f"✅ Skill {state}: {name}"))
            self.refresh_skills_status()
            return
        self.status_label.setText(self.tr(f"❌ 스킬 설정 실패: {name}", f"❌ Skill update failed: {name}"))
        self.append_system_line(
            f"<b>{self.tr('❌ 스킬 설정 실패', '❌ Skill update failed')}</b><br>{html.escape(out)}"
        )

    def install_skill_from_slug(self):
        slug = self.skill_slug_input.text().strip()
        if not slug:
            self.status_label.setText(self.tr("❌ 설치할 slug를 입력하세요", "❌ Enter a slug to install"))
            return
        ok_install, out_install = self.run_shell_command(
            ["clawhub", "install", slug, "--workdir", str(self.gateway_manager.repo_root), "--dir", "skills"],
            cwd=self.gateway_manager.repo_root,
            timeout=180,
        )
        if not ok_install:
            self.status_label.setText(self.tr("❌ ClawHub 설치 실패", "❌ ClawHub install failed"))
            self.append_system_line(
                f"<b>{self.tr('❌ ClawHub 설치 실패', '❌ ClawHub install failed')}</b><br>{html.escape(out_install)}"
            )
            return

        skill_name = slug.split("/")[-1].strip()
        if skill_name:
            self.run_openclaw_command(
                ["config", "set", "--json", f"skills.entries.{skill_name}.enabled", "true"]
            )
        self.status_label.setText(self.tr(f"✅ ClawHub 설치 완료: {slug}", f"✅ ClawHub installed: {slug}"))
        self.refresh_skills_status()

    def persist_panel_visibility(self):
        try:
            settings = self.refresh_settings_cache()
            ui_settings = settings.get("ui", {}) if isinstance(settings.get("ui"), dict) else {}
            ui_settings["leftPanelVisible"] = bool(self.left_panel_visible)
            ui_settings["rightPanelVisible"] = bool(self.right_panel_visible)
            settings["ui"] = ui_settings
            save_settings(settings)
            self.settings_cache = settings
        except Exception as e:
            self.append_debug(f"패널 가시성 저장 실패: {e}")

    def sync_window_width_for_panels(self):
        target_min = 760
        if self.left_panel_visible and self.right_panel_visible:
            target_min = 1280
        elif self.left_panel_visible or self.right_panel_visible:
            target_min = 980
        self.setMinimumWidth(target_min)
        if self.width() < target_min:
            self.resize(target_min, self.height())

    def update_side_panel_visibility(self):
        if hasattr(self, "left_tools_panel"):
            self.left_tools_panel.setVisible(self.left_panel_visible)
        if hasattr(self, "right_tools_panel"):
            self.right_tools_panel.setVisible(self.right_panel_visible)
        if hasattr(self, "left_panel_action"):
            self.left_panel_action.blockSignals(True)
            self.left_panel_action.setChecked(self.left_panel_visible)
            self.left_panel_action.blockSignals(False)
        if hasattr(self, "right_panel_action"):
            self.right_panel_action.blockSignals(True)
            self.right_panel_action.setChecked(self.right_panel_visible)
            self.right_panel_action.blockSignals(False)
        self.sync_window_width_for_panels()

    def on_toggle_left_panel(self, checked: bool):
        self.left_panel_visible = bool(checked)
        self.update_side_panel_visibility()
        self.persist_panel_visibility()

    def on_toggle_right_panel(self, checked: bool):
        self.right_panel_visible = bool(checked)
        self.update_side_panel_visibility()
        self.persist_panel_visibility()

    def show_options_menu(self):
        if not hasattr(self, "options_menu"):
            return
        self.update_side_panel_visibility()
        if hasattr(self, "options_button"):
            anchor = self.options_button.mapToGlobal(QPoint(0, self.options_button.height()))
            self.options_menu.exec(anchor)
            return
        self.options_menu.exec(self.mapToGlobal(QPoint(12, self._header_drag_height)))

    def enable_window_mouse_controls(self):
        self.setMouseTracking(True)
        self.installEventFilter(self)
        if self.centralWidget():
            for widget in self.centralWidget().findChildren(QWidget):
                widget.setMouseTracking(True)
                widget.installEventFilter(self)

    def event_filter_positions(self, obj: QWidget, event) -> tuple[QPoint, QPoint]:
        if hasattr(event, "position"):
            local_on_obj = event.position().toPoint()
        else:
            local_on_obj = event.pos()
        global_pos = obj.mapToGlobal(local_on_obj)
        local_on_window = self.mapFromGlobal(global_pos)
        return global_pos, local_on_window

    def begin_resize(self, edges: dict, global_pos: QPoint) -> bool:
        self._is_resizing = True
        self._resize_edges = edges
        self._resize_start_geom = self.geometry()
        self._resize_start_pos = global_pos
        return True

    def handle_window_pointer_press(self, source_widget: QWidget, event) -> bool:
        if event.button() != Qt.LeftButton:
            return False
        global_pos, local_pos = self.event_filter_positions(source_widget, event)
        edges = self.detect_resize_edges(local_pos)
        if any(edges.values()):
            return self.begin_resize(edges, global_pos)

        interactive_widgets = (QLineEdit, QTextEdit, QPushButton, QComboBox, QSlider, QCheckBox)
        if local_pos.y() <= self._header_drag_height and not isinstance(source_widget, interactive_widgets):
            self._is_dragging = True
            self.drag_position = global_pos - self.frameGeometry().topLeft()
            return True
        return False

    def handle_window_pointer_move(self, source_widget: QWidget, event) -> bool:
        global_pos, local_pos = self.event_filter_positions(source_widget, event)
        if self._is_resizing and (event.buttons() & Qt.LeftButton):
            delta = global_pos - self._resize_start_pos
            rect = QRect(self._resize_start_geom)
            min_w = self.minimumWidth()
            min_h = self.minimumHeight()

            if self._resize_edges.get("left"):
                new_width = self._resize_start_geom.width() - delta.x()
                new_width = max(new_width, min_w)
                rect.setLeft(self._resize_start_geom.right() - new_width)
                rect.setWidth(new_width)
            if self._resize_edges.get("right"):
                new_width = self._resize_start_geom.width() + delta.x()
                rect.setWidth(max(new_width, min_w))
            if self._resize_edges.get("top"):
                new_height = self._resize_start_geom.height() - delta.y()
                new_height = max(new_height, min_h)
                rect.setTop(self._resize_start_geom.bottom() - new_height)
                rect.setHeight(new_height)
            if self._resize_edges.get("bottom"):
                new_height = self._resize_start_geom.height() + delta.y()
                rect.setHeight(max(new_height, min_h))

            self.setGeometry(rect)
            return True

        if self._is_dragging and (event.buttons() & Qt.LeftButton):
            self.move(global_pos - self.drag_position)
            return True

        self.update_resize_cursor(self.detect_resize_edges(local_pos))
        return False

    def handle_window_pointer_release(self, source_widget: QWidget, event) -> bool:
        was_active = self._is_dragging or self._is_resizing
        self._is_dragging = False
        self._is_resizing = False
        self._resize_edges = {"left": False, "right": False, "top": False, "bottom": False}
        _, local_pos = self.event_filter_positions(source_widget, event)
        self.update_resize_cursor(self.detect_resize_edges(local_pos))
        return was_active

    def openclaw_cli(self) -> Path:
        return self.gateway_manager.openclaw_dir / "openclaw.mjs"

    def get_effective_session_key(self) -> str:
        agent_id = self.get_active_agent_id()
        configured = str(self.profile_settings.get("sessionKey") or "").strip()
        if configured and configured.lower() not in LEGACY_SESSION_KEYS:
            return build_agent_session_key(agent_id, configured)
        return build_agent_session_key(agent_id, self.runtime_session_key)

    def run_openclaw_command(self, args: list[str], timeout: int = 25) -> tuple[bool, str]:
        openclaw_mjs = self.openclaw_cli()
        if not openclaw_mjs.exists():
            return False, f"openclaw.mjs not found: {openclaw_mjs}"
        cmd = ["node", str(openclaw_mjs), *args]
        try:
            return_code, stdout_text, stderr_text = run_process_capture(
                cmd,
                cwd=str(self.gateway_manager.openclaw_dir),
                timeout=timeout,
            )
            output = stdout_text.strip() or stderr_text.strip()
            return return_code == 0, output
        except Exception as e:
            return False, str(e)

    def run_shell_command(
        self,
        args: list[str],
        cwd: Optional[Path] = None,
        timeout: int = 60,
    ) -> tuple[bool, str]:
        try:
            return_code, stdout_text, stderr_text = run_process_capture(
                args,
                cwd=str(cwd) if cwd else None,
                timeout=timeout,
            )
            output = stdout_text.strip() or stderr_text.strip()
            return return_code == 0, output
        except Exception as e:
            return False, str(e)

    def handle_local_admin_command(self, message: str) -> bool:
        raw = (message or "").strip()
        if not raw.startswith("/"):
            return False

        lower = raw.lower()
        if lower in {"/session status", "/sess status"}:
            key = self.get_effective_session_key()
            self.append_system_line(f"<b>👻:</b> 현재 sessionKey: <code>{key}</code>")
            self.append_system_line(
                "고정 키를 쓰려면 settings.json의 sessionKey를 명시하고, "
                "기본값(legacy main/default)은 런타임 키로 자동 분리됩니다."
            )
            return True

        if lower in {"/session reset", "/session new", "/sess reset", "/sess new"}:
            self.runtime_session_key = generate_runtime_session_key()
            self.chat_history.clear()
            self.append_system_line(f"<b>👻:</b> 새 세션으로 전환했습니다: <code>{self.runtime_session_key}</code>")
            self.append_system_line(
                "기존 누적 컨텍스트를 분리했습니다. 다음 메시지부터 새 세션으로 처리합니다."
            )
            return True

        if lower in {"/session-memory status", "/sm status"}:
            base_path = self.resolve_memory_search_base_path()
            keys = [
                f"{base_path}.enabled",
                f"{base_path}.sources",
                f"{base_path}.sync.onSessionStart",
            ]
            self.append_system_line("<b>👻:</b> 세션 메모리 설정 상태를 확인합니다...")
            for key in keys:
                ok, out = self.run_openclaw_command(["config", "get", key])
                if ok:
                    self.append_system_line(f"<b>• {key}</b>\n{out}")
                else:
                    self.append_system_line(f"<b>• {key}</b>\n(미설정 또는 조회실패) {out}")
            return True

        if lower in {"/session-memory setup", "/sm setup"}:
            base_path = self.resolve_memory_search_base_path()
            self.append_system_line("<b>👻:</b> 세션 메모리 설정을 적용합니다...")
            commands = [
                ["config", "set", "--json", f"{base_path}.enabled", "true"],
                [
                    "config",
                    "set",
                    "--json",
                    f"{base_path}.sources",
                    "[\"memory\",\"sessions\"]",
                ],
                [
                    "config",
                    "set",
                    "--json",
                    f"{base_path}.sync.onSessionStart",
                    "true",
                ],
            ]
            all_ok = True
            for args in commands:
                ok, out = self.run_openclaw_command(args)
                all_ok = all_ok and ok
                line = "✅" if ok else "❌"
                self.append_system_line(f"{line} {' '.join(args)}\n{out}")
            if all_ok:
                self.append_system_line(
                    "<b>완료:</b> 세션 메모리 설정 적용됨. 게이트웨이 재시작 후 반영됩니다."
                )
            else:
                self.append_system_line("<b>경고:</b> 일부 설정 적용 실패. 위 로그를 확인하세요.")
            return True

        if lower in {"/session-memory off", "/sm off"}:
            base_path = self.resolve_memory_search_base_path()
            self.append_system_line("<b>👻:</b> 세션 메모리 기능을 비활성화합니다...")
            ok, out = self.run_openclaw_command(
                ["config", "set", "--json", f"{base_path}.enabled", "false"]
            )
            if ok:
                self.append_system_line(f"✅ {out}\n게이트웨이 재시작 후 반영됩니다.")
            else:
                self.append_system_line(f"❌ {out}")
            return True

        # Skill management (local, deterministic)
        if lower in {"/skill list", "/skills list", "/skill status", "/skills status"}:
            self.append_system_line("<b>👻:</b> 스킬 상태를 조회합니다...")
            ok1, out1 = self.run_openclaw_command(["skills", "list"])
            ok2, out2 = self.run_openclaw_command(["config", "get", "skills.entries"])
            if ok1:
                self.append_system_line(f"<b>OpenClaw skills list</b>\n{out1}")
            else:
                self.append_system_line(f"❌ skills list 실패\n{out1}")
            if ok2:
                self.append_system_line(f"<b>skills.entries</b>\n{out2}")
            else:
                self.append_system_line(f"<b>skills.entries</b>\n(미설정 또는 조회실패) {out2}")
            return True

        if lower.startswith("/skill enable ") or lower.startswith("/skill register "):
            parts = shlex.split(raw)
            if len(parts) < 3:
                self.append_system_line("❌ 사용법: /skill enable <name>")
                return True
            skill_name = parts[2].strip()
            if not skill_name:
                self.append_system_line("❌ 스킬 이름이 비어 있습니다.")
                return True
            ok, out = self.run_openclaw_command(
                ["config", "set", "--json", f"skills.entries.{skill_name}.enabled", "true"]
            )
            if ok:
                self.append_system_line(f"✅ 스킬 활성화: {skill_name}\n{out}\n게이트웨이 재시작 후 반영됩니다.")
            else:
                self.append_system_line(f"❌ 스킬 활성화 실패: {skill_name}\n{out}")
            return True

        if lower.startswith("/skill disable ") or lower.startswith("/skill unregister "):
            parts = shlex.split(raw)
            if len(parts) < 3:
                self.append_system_line("❌ 사용법: /skill disable <name>")
                return True
            skill_name = parts[2].strip()
            if not skill_name:
                self.append_system_line("❌ 스킬 이름이 비어 있습니다.")
                return True
            ok, out = self.run_openclaw_command(
                ["config", "set", "--json", f"skills.entries.{skill_name}.enabled", "false"]
            )
            if ok:
                self.append_system_line(f"✅ 스킬 비활성화: {skill_name}\n{out}\n게이트웨이 재시작 후 반영됩니다.")
            else:
                self.append_system_line(f"❌ 스킬 비활성화 실패: {skill_name}\n{out}")
            return True

        if lower.startswith("/skill install "):
            parts = shlex.split(raw)
            if len(parts) < 3:
                self.append_system_line("❌ 사용법: /skill install <slug>")
                return True
            slug = parts[2].strip()
            if not slug:
                self.append_system_line("❌ slug가 비어 있습니다.")
                return True
            self.append_system_line(f"<b>👻:</b> ClawHub 설치 시도: {slug}")
            ok_install, out_install = self.run_shell_command(
                ["clawhub", "install", slug, "--workdir", str(self.gateway_manager.repo_root), "--dir", "skills"],
                cwd=self.gateway_manager.repo_root,
                timeout=180,
            )
            if not ok_install:
                self.append_system_line(f"❌ 설치 실패: {slug}\n{out_install}")
                return True
            self.append_system_line(f"✅ 설치 완료: {slug}\n{out_install}")
            # slug가 owner/name 형식일 수 있으므로 마지막 토큰을 config 엔트리 키로 사용
            skill_name = slug.split("/")[-1].strip()
            if skill_name:
                ok_enable, out_enable = self.run_openclaw_command(
                    ["config", "set", "--json", f"skills.entries.{skill_name}.enabled", "true"]
                )
                if ok_enable:
                    self.append_system_line(
                        f"✅ 자동 활성화: {skill_name}\n{out_enable}\n게이트웨이 재시작 후 반영됩니다."
                    )
                else:
                    self.append_system_line(
                        f"⚠️ 설치는 성공했지만 자동 활성화 실패: {skill_name}\n{out_enable}"
                    )
            return True

        return False

    def remove_last_nonempty_block(self):
        doc = self.chat_area.document()
        block = doc.lastBlock()
        if not block.isValid():
            return
        if block.text().strip() == "" and block.previous().isValid():
            block = block.previous()
        cursor = QTextCursor(block)
        cursor.select(QTextCursor.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deleteChar()

    def clear_stream_preview(self):
        if not self.stream_preview_active:
            return
        if self.stream_preview_start_pos is None:
            self.remove_last_nonempty_block()
            self.stream_preview_active = False
            return
        doc = self.chat_area.document()
        end_pos = max(doc.characterCount() - 1, self.stream_preview_start_pos)
        cursor = QTextCursor(doc)
        cursor.setPosition(self.stream_preview_start_pos)
        cursor.setPosition(end_pos, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        self.stream_preview_active = False
        self.stream_preview_start_pos = None

    def remove_processing_line_if_present(self):
        doc = self.chat_area.document()
        block = doc.lastBlock()
        if not block.isValid():
            return
        if block.text().strip() == "" and block.previous().isValid():
            block = block.previous()
        if "처리 중..." in block.text():
            cursor = QTextCursor(block)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def on_agent_partial(self, partial_text: str):
        text = (partial_text or "").strip()
        if not text:
            return
        if len(text) <= 2 and re.fullmatch(r"[^\w가-힣]+", text):
            return
        if self.collapse_reasoning_check.isChecked():
            # Collapse mode: keep UI clean, only final answer is rendered.
            if self.stream_preview_active:
                self.clear_stream_preview()
            return
        if self.should_collapse_line(text):
            return
        now = time.time()
        if text == self.last_partial_preview and (now - self.last_partial_preview_at) < 0.15:
            return
        if (now - self.last_partial_preview_at) < 0.12:
            return
        self.last_partial_preview = text
        self.last_partial_preview_at = now
        self.remove_processing_line_if_present()
        preview_body = self.text_to_html(text)
        preview = (
            "<div style='margin:8px 0;'>"
            "<div style='color:#8fd8ff; font-weight:600; margin-bottom:2px;'>👻(초안)</div>"
            "<div style='padding:8px 10px; background: rgba(255,255,255,0.03); "
            f"border:1px solid rgba(255,255,255,0.07); border-radius:10px;'>{preview_body}</div></div>"
        )
        self.clear_stream_preview()
        cursor = self.chat_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.stream_preview_start_pos = cursor.position()
        cursor.insertHtml(preview)
        cursor.insertBlock()
        self.chat_area.setTextCursor(cursor)
        self.stream_preview_active = True

    def on_chat_anchor_clicked(self, url):
        link = url.toString() if hasattr(url, "toString") else str(url)
        if not link.startswith("reason://"):
            return
        reasoning_id = link.replace("reason://", "", 1).strip()
        reasoning_text = self.hidden_reasoning_store.get(reasoning_id, "").strip()
        if not reasoning_text:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self.tr("접힌 중간 과정", "Folded Reasoning"))
        dialog.resize(760, 520)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        note = QLabel(
            self.tr(
                "사고접기 ON 상태에서 숨겨진 중간 출력입니다.",
                "Hidden intermediate output while collapse mode is ON.",
            )
        )
        note.setStyleSheet("color:#9ca3af; font-size:11px;")
        layout.addWidget(note)

        viewer = QTextEdit()
        viewer.setReadOnly(True)
        viewer.setPlainText(reasoning_text)
        viewer.setStyleSheet(
            "QTextEdit { background:#0f141b; color:#e5e7eb; border:1px solid #2a2f3a; "
            "border-radius:6px; font-family:'Consolas','D2Coding','Malgun Gothic',monospace; font-size:12px; }"
        )
        layout.addWidget(viewer, 1)

        close_btn = QPushButton(self.tr("닫기", "Close"))
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec()
        
    def init_tray(self):
        """시스템 트레이 아이콘 설정"""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
            
        self.tray = QSystemTrayIcon(self)
        # 기본 아이콘 설정 (경고 방지)
        try:
            self.tray.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        except Exception:
            pass
        self.tray.setToolTip("Ghost Chat")
        
        menu = QMenu()
        
        show_action = QAction("보이기", self)
        show_action.triggered.connect(self.show)
        menu.addAction(show_action)
        
        hide_action = QAction("숨기기", self)
        hide_action.triggered.connect(self.hide)
        menu.addAction(hide_action)

        self.always_on_top_action = QAction("항상 위 고정", self)
        self.always_on_top_action.setCheckable(True)
        self.always_on_top_action.setChecked(self.always_on_top)
        self.always_on_top_action.toggled.connect(self.on_always_on_top_toggled)
        menu.addAction(self.always_on_top_action)
        
        menu.addSeparator()

        quit_action = QAction("종료", self)
        quit_action.triggered.connect(self.quit_window)
        menu.addAction(quit_action)
        
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self.tray_activated)
        try:
            self.tray.show()
        except Exception:
            pass

    def quit_window(self):
        """트레이 메뉴에서 호출되는 실제 종료"""
        self._force_close = True
        try:
            if hasattr(self, "tray") and self.tray:
                self.tray.hide()
        except Exception:
            pass
        self.close()

    def get_active_ollama_model_name(self) -> str:
        profile = self.profile_settings if isinstance(self.profile_settings, dict) else {}
        source = str(profile.get("modelSource") or "").strip().lower()
        if source != "ollama":
            return ""
        ollama = profile.get("ollama", {}) if isinstance(profile.get("ollama"), dict) else {}
        model_name = str(ollama.get("modelName") or "").strip()
        if model_name:
            return model_name
        model_ref = str(profile.get("model") or "").strip()
        if model_ref.startswith("ollama/"):
            return model_ref.split("/", 1)[1].strip()
        return ""

    def unload_ollama_model_on_close(self):
        model_name = self.get_active_ollama_model_name()
        if not model_name:
            return
        ollama_cli = resolve_ollama_cli_path()
        if not ollama_cli:
            logger.warning(f"ollama cli not found while unloading model: {model_name}")
            return
        code, stdout_text, stderr_text = run_process_capture(
            [ollama_cli, "stop", model_name],
            timeout=8,
        )
        if code == 0:
            logger.info(f"ollama model unloaded on chat close: {model_name}")
            return
        detail = stderr_text.strip() or stdout_text.strip()
        logger.warning(f"failed to unload ollama model on close: {model_name} / {detail}")
        
    def tray_activated(self, reason):
        """트레이 아이콘 클릭 시"""
        if reason == QSystemTrayIcon.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show()
    
    def startup_sequence(self):
        """시작 시퀀스: 게이트웨이 + 온보드"""
        self.refresh_settings_cache()
        self.gateway_manager.settings = self.profile_settings
        self.request_profile_runtime_sync(announce=False)
        running = self.gateway_manager.check_gateway_running()
        if running:
            self.on_startup_ready()
            self.append_system_line("✅ 기존 게이트웨이 감지: 즉시 연결합니다.")
        else:
            self.ready = False
            self.status_label.setText("⛔ 게이트웨이 오프라인")
            self.btn_send.setEnabled(False)
            self.input_line.setEnabled(False)
            self.append_system_line(
                "⚠️ 게이트웨이가 꺼져 있습니다. GhostRelay Console에서 시작 버튼으로 켜주세요."
            )

        self.gateway_watch_timer = QTimer(self)
        self.gateway_watch_timer.setInterval(1500)
        self.gateway_watch_timer.timeout.connect(self.poll_gateway_status)
        self.gateway_watch_timer.start()

    def poll_gateway_status(self):
        running = self.gateway_manager.check_gateway_running()
        desired_interval = 1500 if running else 3000
        if self.gateway_watch_timer.interval() != desired_interval:
            self.gateway_watch_timer.setInterval(desired_interval)
        if running and not self.ready:
            self.on_startup_ready()
            self.append_system_line("✅ 게이트웨이 연결됨. 채팅을 시작하세요.")
            return
        if not running and self.ready:
            self.ready = False
            self.status_label.setText("⛔ 게이트웨이 오프라인")
            self.btn_send.setEnabled(False)
            self.input_line.setEnabled(False)
            self.append_system_line(
                "⚠️ 게이트웨이 연결이 끊겼습니다. GhostRelay Console에서 재시작하세요."
            )

    def on_startup_ready(self):
        self.ready = True
        self.status_label.setText("✅ 준비 완료")
        self.btn_send.setEnabled(True)
        self.input_line.setEnabled(True)

    def on_startup_failed(self, detail: str):
        self.ready = False
        self.status_label.setText("❌ 게이트웨이 시작 실패")
        self.append_system_line("❌ 게이트웨이 시작에 실패했습니다.")
        self.append_system_line("💡 로그 파일을 확인하거나 수동으로 시작해주세요:")
        self.append_system_line(f"   {self.gateway_manager.gateway_bat}")
        if detail:
            self.append_debug(f"시작 실패 상세: {detail}")

    def update_processing_indicator(self):
        if not self.processing_started_at:
            return
        elapsed = time.time() - self.processing_started_at
        dots = "." * ((self.processing_step % 3) + 1)
        self.status_label.setText(f"🤔 생각중{dots} {elapsed:.1f}s")
        self.processing_step += 1

    def start_processing_indicator(self):
        self.processing_started_at = time.time()
        self.processing_step = 0
        if not self.processing_timer.isActive():
            self.processing_timer.start()

    def stop_processing_indicator(self):
        if self.processing_timer.isActive():
            self.processing_timer.stop()
        self.processing_started_at = None
        if self.ready:
            self.status_label.setText("✅ 준비 완료")
        
    def send_message(self):
        """메시지 전송"""
        if not self.ready:
            self.chat_area.append("⚠️ 아직 준비되지 않았습니다. 잠시만 기다려주세요...")
            return
        
        message = self.input_line.text().strip()
        if not message:
            return
        
        if self.agent_thread and self.agent_thread.isRunning():
            self.chat_area.append("⏳ 이전 요청 처리 중...")
            return

        # 로컬 관리 명령 우선 처리 (모델 tool-loop 방지)
        if self.handle_local_admin_command(message):
            self.input_line.clear()
            self.chat_area.moveCursor(QTextCursor.End)
            return
        
        # 입력창 비활성화
        self.input_line.clear()
        self.input_line.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.start_processing_indicator()
        self.last_partial_preview = ""
        self.last_partial_preview_at = 0.0
        if hasattr(self, "runtime_stats_label"):
            self.runtime_stats_label.setText("Ctx 계산 중...")
            self.runtime_stats_label.setToolTip("요청 처리 후 런타임 지표를 갱신합니다.")
        
        # 채팅 영역에 표시
        user_html = (
            "<div style='margin: 8px 0;'>"
            "<div style='color:#9be29f; font-weight:600; margin-bottom:2px;'>나</div>"
            f"<div style='padding:8px 10px; background: rgba(107,208,255,0.08); "
            f"border:1px solid rgba(107,208,255,0.25); border-radius:10px;'>{self.text_to_html(message)}</div>"
            "</div>"
        )
        self.chat_area.append(user_html)
        self.chat_area.append("🤔 처리 중...")
        self.chat_area.moveCursor(QTextCursor.End)
        
        # Agent 실행
        self.refresh_settings_cache()
        self.gateway_manager.settings = self.profile_settings
        source = str(self.profile_settings.get("modelSource") or "").strip().lower()
        tool_mode = str(self.profile_settings.get("toolMode") or "auto").strip().lower()
        ollama_cfg = self.profile_settings.get("ollama", {}) if isinstance(self.profile_settings.get("ollama"), dict) else {}
        supports_tools = ollama_cfg.get("supportsTools") if isinstance(ollama_cfg.get("supportsTools"), bool) else None
        is_ollama_chat_mode = (
            source == "ollama"
            and (
                tool_mode == "chat"
                or (tool_mode == "auto" and supports_tools is not True)
            )
        )
        gateway_token = "" if is_ollama_chat_mode else self.gateway_manager.get_gateway_token()
        thinking_mode = self.thinking_combo.currentText().strip() or "off"
        agent_id = self.get_active_agent_id()
        session_key = self.get_effective_session_key()
        self.chat_history.append({"role": "user", "content": message})
        self.agent_thread = OpenClawAgentThread(
            message,
            gateway_token,
            list(self.chat_history),
            session_key=session_key,
            agent_id=agent_id,
            profile_settings=dict(self.profile_settings),
            thinking_mode=thinking_mode,
        )
        self.agent_thread.message_received.connect(self.on_agent_response)
        self.agent_thread.debug_received.connect(self.append_debug)
        self.agent_thread.partial_received.connect(self.on_agent_partial)
        self.agent_thread.stats_received.connect(self.on_agent_stats)
        self.agent_thread.finished.connect(self.on_agent_finished)
        self.agent_thread.start()
        
    def on_agent_response(self, message: str):
        """Agent 응답 수신"""
        self.remove_processing_line_if_present()
        if self.stream_preview_active:
            self.clear_stream_preview()

        normalized, latency, hidden_reasoning = self.normalize_agent_text(message)
        if not normalized:
            normalized = self.tr(
                "유효한 응답이 없어 다시 시도해 주세요.",
                "No valid response was returned. Please try again.",
            )
        hidden_reasoning_id = None
        hidden_reasoning_lines = 0
        if self.collapse_reasoning_check.isChecked():
            hidden_reasoning_id, hidden_reasoning_lines = self.register_hidden_reasoning(hidden_reasoning)
        rendered = self.render_assistant_block(
            normalized,
            latency,
            hidden_reasoning_id=hidden_reasoning_id,
            hidden_reasoning_lines=hidden_reasoning_lines,
        )
        self.chat_area.append(rendered)
        history_text = normalized
        if latency is not None:
            history_text = f"{normalized}\n\n(응답 {latency:.1f}초)"
        self.chat_history.append({"role": "assistant", "content": history_text})
        self.chat_area.moveCursor(QTextCursor.End)
        
    def on_agent_finished(self):
        """Agent 실행 완료"""
        self.stop_processing_indicator()
        self.input_line.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.stream_preview_active = False
        self.stream_preview_start_pos = None
        self.last_partial_preview = ""
        self.last_partial_preview_at = 0.0

    def detect_resize_edges(self, pos: QPoint) -> dict:
        w = self.width()
        h = self.height()
        m = self._resize_margin
        return {
            "left": pos.x() <= m,
            "right": pos.x() >= w - m,
            "top": pos.y() <= m,
            "bottom": pos.y() >= h - m,
        }

    def update_resize_cursor(self, edges: dict):
        left, right, top, bottom = (
            edges.get("left"),
            edges.get("right"),
            edges.get("top"),
            edges.get("bottom"),
        )
        if (left and top) or (right and bottom):
            self.setCursor(Qt.SizeFDiagCursor)
        elif (right and top) or (left and bottom):
            self.setCursor(Qt.SizeBDiagCursor)
        elif left or right:
            self.setCursor(Qt.SizeHorCursor)
        elif top or bottom:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.unsetCursor()
        
    def mousePressEvent(self, event):
        """드래그 시작"""
        if self.handle_window_pointer_press(self, event):
            event.accept()
            return
        super().mousePressEvent(event)
             
    def mouseMoveEvent(self, event):
        """드래그/리사이즈"""
        if self.handle_window_pointer_move(self, event):
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.handle_window_pointer_release(self, event)
        super().mouseReleaseEvent(event)

    def eventFilter(self, obj, event):
        if not isinstance(obj, QWidget):
            return super().eventFilter(obj, event)
        if obj.window() is not self:
            return super().eventFilter(obj, event)

        event_type = event.type()
        if event_type == QEvent.MouseButtonPress:
            if self.handle_window_pointer_press(obj, event):
                event.accept()
                return True
        elif event_type == QEvent.MouseMove:
            if self.handle_window_pointer_move(obj, event):
                event.accept()
                return True
        elif event_type == QEvent.MouseButtonRelease:
            if self.handle_window_pointer_release(obj, event):
                event.accept()
                return True
        return super().eventFilter(obj, event)
            
    def closeEvent(self, event):
        """창 닫기 시 트레이로 최소화"""
        tray_visible = bool(hasattr(self, "tray") and self.tray and self.tray.isVisible())
        if (tray_visible or self.console_managed) and not self._force_close:
            self.hide()
            event.ignore()
        else:
            self.unload_ollama_model_on_close()
            if self.agent_thread and self.agent_thread.isRunning():
                self.agent_thread.terminate()
                self.agent_thread.wait(1000)
            try:
                if hasattr(self, "gateway_watch_timer") and self.gateway_watch_timer:
                    self.gateway_watch_timer.stop()
            except Exception:
                pass
            try:
                if hasattr(self, "processing_timer") and self.processing_timer:
                    self.processing_timer.stop()
            except Exception:
                pass
            try:
                if hasattr(self, "tray") and self.tray:
                    self.tray.hide()
                    self.tray.deleteLater()
                    self.tray = None
            except Exception:
                pass
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    
    window = GhostChatWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

