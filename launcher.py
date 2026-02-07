"""
Ghost Chat Launcher
시스템 트레이 + 툴 모음집
"""

import sys
import subprocess
import webbrowser
import time
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSystemTrayIcon, QMenu, QLabel, QStyle, QDialog,
    QLineEdit, QComboBox, QFileDialog, QMessageBox, QInputDialog, QCheckBox, QSlider,
    QPlainTextEdit
)
from PySide6.QtCore import Qt, QPoint, QThread, Signal, QTimer
from PySide6.QtGui import QAction

sys.path.insert(0, str(Path(__file__).parent))
from runtime_paths import resolve_gateway_script, resolve_openclaw_dir
from settings_store import (
    default_profile,
    default_workspace_for_profile,
    get_active_profile_id,
    get_profile,
    list_profiles,
    load_settings,
    normalize_agent_id,
    normalize_provider_id,
    normalize_profile_id,
    save_settings,
)

OPENCLAW_MIN_CONTEXT_WINDOW = 16000
DEFAULT_OLLAMA_CONTEXT_WINDOW = 32768
MAX_OLLAMA_CONTEXT_WINDOW = 262144
DEFAULT_API_CONTEXT_WINDOW = 32768
MAX_API_CONTEXT_WINDOW = 262144


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
    subprocess_kwargs = {}
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
    return (
        result.returncode,
        decode_process_output(result.stdout),
        decode_process_output(result.stderr),
    )


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


def check_gateway_running(port: int = 18789) -> bool:
    try:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock_v4:
            # UI 폴링에서 멈춤을 줄이기 위해 localhost IPv4만 빠르게 확인한다.
            sock_v4.settimeout(0.05)
            return sock_v4.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False


def wait_gateway_state(
    target_running: bool,
    timeout_sec: float = 30.0,
    interval_sec: float = 0.35,
    on_tick=None,
) -> bool:
    started = time.time()
    deadline = started + timeout_sec
    while time.time() < deadline:
        running = check_gateway_running()
        if running == target_running:
            return True
        if callable(on_tick):
            try:
                on_tick(time.time() - started)
            except Exception:
                pass
        time.sleep(interval_sec)
    if callable(on_tick):
        try:
            on_tick(timeout_sec)
        except Exception:
            pass
    return check_gateway_running() == target_running


def list_listening_pids(port: int = 18789) -> list[int]:
    if sys.platform != "win32":
        return []
    ps_query = (
        "Get-NetTCPConnection -LocalPort "
        f"{port} -State Listen -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty OwningProcess -Unique"
    )
    code_ps, stdout_ps, _ = run_process_capture(
        ["powershell", "-NoProfile", "-Command", ps_query],
        timeout=6,
    )
    if code_ps == 0 and stdout_ps.strip():
        pids_ps: set[int] = set()
        for line in stdout_ps.splitlines():
            raw = line.strip()
            if raw.isdigit():
                pids_ps.add(int(raw))
        if pids_ps:
            return sorted(pids_ps)

    code, stdout_text, _ = run_process_capture(
        ["netstat", "-ano", "-p", "tcp"],
        timeout=6,
    )
    if code != 0 and not stdout_text.strip():
        return []
    pids: set[int] = set()
    for line in stdout_text.splitlines():
        raw_line = line.strip()
        if not raw_line:
            continue
        parts = raw_line.split()
        if len(parts) < 5:
            continue
        local_addr = parts[1]
        if not re.search(rf":{port}$", local_addr):
            continue
        state = " ".join(parts[3:-1]).upper()
        # Windows locale-safe listen checks (e.g. LISTENING / 수신 대기)
        if "LISTEN" not in state and "수신" not in state:
            continue
        try:
            pid = int(parts[-1])
        except Exception:
            continue
        pids.add(pid)
    return sorted(pids)


def kill_pids(pids: list[int]) -> tuple[bool, str]:
    ok = True
    errors: list[str] = []
    for pid in pids:
        code, stdout_text, stderr_text = run_process_capture(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            timeout=8,
        )
        if code != 0:
            ok = False
            detail = (stderr_text or stdout_text or "").strip()
            if not detail:
                detail = "taskkill failed"
            errors.append(f"PID {pid}: {detail}")
    return ok, "; ".join(errors)


def get_ollama_tags_url(base_url: str) -> str:
    raw_base = (base_url or "").strip().rstrip("/")
    if not raw_base:
        raw_base = "http://127.0.0.1:11434/v1"
    if raw_base.endswith("/v1"):
        raw_base = raw_base[:-3]
    return f"{raw_base}/api/tags"


def fetch_ollama_models(base_url: str) -> list[str]:
    import urllib.request

    tags_url = get_ollama_tags_url(base_url)
    with urllib.request.urlopen(tags_url, timeout=5) as resp:
        if getattr(resp, "status", 200) >= 400:
            raise RuntimeError(f"Ollama status {resp.status}")
        payload = resp.read().decode("utf-8", errors="ignore")
    data = json.loads(payload or "{}")
    models_raw = data.get("models") or []
    names = []
    if isinstance(models_raw, list):
        for item in models_raw:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                if name:
                    names.append(name)
    return sorted(set(names))


def get_openai_models_url(base_url: str) -> str:
    raw_base = (base_url or "").strip().rstrip("/")
    if not raw_base:
        raw_base = "https://api.openai.com/v1"
    if raw_base.endswith("/models"):
        return raw_base
    lowered = raw_base.lower()
    if lowered.endswith("/openai"):
        return f"{raw_base}/models"
    if raw_base.endswith("/v1"):
        return f"{raw_base}/models"
    if raw_base.endswith("/v1beta"):
        return f"{raw_base}/models"
    return f"{raw_base}/v1/models"


def fetch_openai_compatible_models(base_url: str, api_key: str) -> list[str]:
    import urllib.request
    import urllib.error

    models_url = get_openai_models_url(base_url)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(models_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = str(e)
        raise RuntimeError(f"HTTP {e.code}: {body.strip() or str(e)}")
    except Exception as e:
        raise RuntimeError(str(e))

    data = json.loads(payload or "{}")
    rows = data.get("data") if isinstance(data, dict) else []
    model_ids: list[str] = []
    if isinstance(rows, list):
        for item in rows:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if model_id:
                model_ids.append(model_id)
    return sorted(set(model_ids))


def get_anthropic_models_url(base_url: str) -> str:
    raw_base = (base_url or "").strip().rstrip("/")
    if not raw_base:
        raw_base = "https://api.anthropic.com/v1"
    if raw_base.endswith("/models"):
        return raw_base
    if raw_base.endswith("/v1"):
        return f"{raw_base}/models"
    return f"{raw_base}/v1/models"


def fetch_anthropic_models(base_url: str, api_key: str) -> list[str]:
    import urllib.request
    import urllib.error

    models_url = get_anthropic_models_url(base_url)
    headers = {
        "Accept": "application/json",
        "x-api-key": str(api_key or "").strip(),
        "anthropic-version": "2023-06-01",
    }
    req = urllib.request.Request(models_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = str(e)
        raise RuntimeError(f"HTTP {e.code}: {body.strip() or str(e)}")
    except Exception as e:
        raise RuntimeError(str(e))

    data = json.loads(payload or "{}")
    rows = data.get("data") if isinstance(data, dict) else []
    model_ids: list[str] = []
    if isinstance(rows, list):
        for item in rows:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if model_id:
                model_ids.append(model_id)
    return sorted(set(model_ids))


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
            Path(os.environ.get("APPDATA", "")) / "npm" / "ollama.cmd",
            Path(os.environ.get("LOCALAPPDATA", "")) / "npm" / "ollama.cmd",
            Path(os.environ.get("ProgramFiles", "")) / "nodejs" / "ollama.cmd",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "nodejs" / "ollama.cmd",
        ]
        for candidate in candidates:
            if str(candidate).strip() and candidate.exists():
                return str(candidate)
    return None


def ollama_host_from_base_url(base_url: str) -> str:
    raw = str(base_url or "").strip().rstrip("/")
    if not raw:
        raw = "http://127.0.0.1:11434/v1"
    if raw.endswith("/v1"):
        raw = raw[:-3]
    return raw


def run_ollama_cli_capture(
    args: list[str],
    base_url: str,
    cwd: str | None = None,
    timeout: int = 60,
) -> tuple[int, str, str]:
    ollama_cli = resolve_ollama_cli_path()
    if not ollama_cli:
        return 1, "", "Ollama CLI를 찾지 못했습니다. (OLLAMA_CLI 또는 PATH 확인)"
    command = [ollama_cli, *args]
    if sys.platform == "win32" and Path(ollama_cli).suffix.lower() in {".cmd", ".bat"}:
        command = ["cmd", "/c", ollama_cli, *args]
    env = os.environ.copy()
    host = ollama_host_from_base_url(base_url)
    if host:
        env["OLLAMA_HOST"] = host
    return run_process_capture(command, cwd=cwd, timeout=timeout, env=env)


def parse_ollama_list_output(stdout_text: str) -> list[str]:
    models: list[str] = []
    for line in str(stdout_text or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.lower().startswith("name ") or raw.startswith("NAME "):
            continue
        if set(raw) <= {"-", " "}:
            continue
        model_name = raw.split()[0].strip()
        if model_name:
            models.append(model_name)
    return sorted(set(models))


def fetch_ollama_models_any(base_url: str) -> list[str]:
    try:
        return fetch_ollama_models(base_url)
    except Exception:
        code, stdout_text, stderr_text = run_ollama_cli_capture(["list"], base_url=base_url, timeout=20)
        if code != 0:
            detail = (stderr_text or stdout_text or "").strip() or "ollama list failed"
            raise RuntimeError(detail)
        models = parse_ollama_list_output(stdout_text)
        return models


def sanitize_ollama_model_name(name: str) -> str:
    raw = str(name or "").strip().lower()
    if not raw:
        return ""
    cleaned = re.sub(r"[^a-z0-9._:-]+", "-", raw).strip("-")
    return cleaned


def build_modelfile_text(gguf_file_name: str, temperature: str = "0.7") -> str:
    temp_raw = str(temperature or "0.7").strip()
    try:
        temp_value = float(temp_raw)
    except Exception:
        temp_value = 0.7
    return f"FROM ./{gguf_file_name}\nPARAMETER temperature {temp_value}\n"


def detect_ollama_context_window(base_url: str, model_name: str) -> tuple[Optional[int], str]:
    import urllib.request

    raw_model = str(model_name or "").strip()
    if not raw_model:
        return None, "모델 이름이 비어 있습니다."
    if raw_model.startswith("ollama/"):
        raw_model = raw_model.split("/", 1)[1].strip()
    if not raw_model:
        return None, "모델 이름이 비어 있습니다."

    raw_base = (base_url or "").strip().rstrip("/")
    if not raw_base:
        raw_base = "http://127.0.0.1:11434/v1"
    if raw_base.endswith("/v1"):
        raw_base = raw_base[:-3]
    show_url = f"{raw_base}/api/show"

    payload = json.dumps({"model": raw_model}).encode("utf-8")
    req = urllib.request.Request(
        show_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=6) as resp:
        if getattr(resp, "status", 200) >= 400:
            return None, f"Ollama status {resp.status}"
        raw_text = resp.read().decode("utf-8", errors="ignore")

    data = json.loads(raw_text or "{}")

    def parse_int(value: Any) -> Optional[int]:
        try:
            text = str(value).strip()
            if not text:
                return None
            return int(float(text))
        except Exception:
            return None

    candidates = [
        ("context_window", data.get("context_window")),
        ("context_length", data.get("context_length")),
        ("num_ctx", data.get("num_ctx")),
    ]
    details = data.get("details") if isinstance(data.get("details"), dict) else {}
    candidates.extend(
        [
            ("details.context_window", details.get("context_window")),
            ("details.context_length", details.get("context_length")),
            ("details.num_ctx", details.get("num_ctx")),
        ]
    )
    model_info = data.get("model_info") if isinstance(data.get("model_info"), dict) else {}
    for key, value in model_info.items():
        if "context_length" in key or "context_window" in key or key.endswith(".num_ctx"):
            candidates.append((f"model_info.{key}", value))

    for source, value in candidates:
        parsed = parse_int(value)
        if parsed and parsed > 0:
            return parsed, source

    params_text = str(data.get("parameters") or "")
    param_match = re.search(r"(num_ctx|context_length|context_window)\s+(\d+)", params_text, re.IGNORECASE)
    if param_match:
        parsed = parse_int(param_match.group(2))
        if parsed and parsed > 0:
            return parsed, "parameters"

    return None, "모델 메타데이터에서 컨텍스트 길이를 찾지 못했습니다."


def detect_ollama_tools_support(
    base_url: str,
    model_name: str,
    api_key: str,
) -> tuple[Optional[bool], str]:
    import urllib.request
    import urllib.error

    raw_model = str(model_name or "").strip()
    if not raw_model:
        return None, "모델 이름이 비어 있습니다."
    if raw_model.startswith("ollama/"):
        raw_model = raw_model.split("/", 1)[1].strip()
    if not raw_model:
        return None, "모델 이름이 비어 있습니다."

    raw_base = (base_url or "").strip().rstrip("/")
    if not raw_base:
        raw_base = "http://127.0.0.1:11434/v1"
    if not raw_base.endswith("/v1"):
        raw_base = f"{raw_base}/v1"
    probe_url = f"{raw_base}/chat/completions"

    payload = {
        "model": raw_model,
        "stream": False,
        "messages": [{"role": "user", "content": "tool capability probe"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "probe tool support",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 1,
    }
    headers = {"Content-Type": "application/json"}
    key = str(api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        probe_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            status = int(getattr(resp, "status", 200))
            if status >= 400:
                return None, f"Ollama status {status}"
            _ = resp.read()
        return True, "tool 호출 지원"
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = str(e)
        lowered = body.lower()
        if (
            "does not support tools" in lowered
            or "tools are not supported" in lowered
            or ("tool calling" in lowered and "not support" in lowered)
        ):
            return False, "모델이 tool-calling을 지원하지 않습니다."
        if "unknown model" in lowered:
            return None, "모델 이름을 찾지 못했습니다. Ollama 목록/모델명을 확인하세요."
        return None, f"HTTP {e.code}: {body.strip() or str(e)}"
    except Exception as e:
        return None, str(e)


def build_gateway_env(profile: dict | None = None) -> dict[str, str]:
    env = os.environ.copy()
    selected = profile if isinstance(profile, dict) else {}
    model = str(selected.get("model") or "").strip()
    workspace = str(selected.get("workspace") or "").strip()
    source = str(selected.get("modelSource") or "vertex").strip().lower()

    if model:
        env["OPENCLAW_DEFAULT_MODEL"] = model
    if workspace:
        env["OPENCLAW_WORKSPACE"] = workspace

    if source == "ollama":
        ollama = selected.get("ollama", {}) if isinstance(selected.get("ollama"), dict) else {}
        base_url = str(ollama.get("baseUrl") or "").strip()
        api_key = str(ollama.get("apiKey") or "").strip() or "ollama-local"
        if base_url:
            env["GHOSTRELAY_OLLAMA_BASE_URL"] = base_url
        env["GHOSTRELAY_OLLAMA_API_KEY"] = api_key
        return env

    if source == "api":
        api_provider = (
            selected.get("apiProvider", {})
            if isinstance(selected.get("apiProvider"), dict)
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

    vertex = selected.get("vertex", {}) if isinstance(selected.get("vertex"), dict) else {}
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


PROFILE_FILE_NAMES = [
    "AGENTS.md",
    "SOUL.md",
    "IDENTITY.md",
    "USER.md",
    "TOOLS.md",
    "HEARTBEAT.md",
    "MEMORY.md",
]


def default_profile_file_content(name: str, profile_id: str) -> str:
    normalized = normalize_profile_id(profile_id)
    templates = {
        "AGENTS.md": f"# AGENTS ({normalized})\n\n- Add operating rules for this profile.\n",
        "SOUL.md": f"# SOUL ({normalized})\n\n- Define tone, style, and behavior boundaries.\n",
        "IDENTITY.md": f"# IDENTITY ({normalized})\n\n- Define assistant identity for this profile.\n",
        "USER.md": f"# USER ({normalized})\n\n- Define user preferences and restrictions.\n",
        "TOOLS.md": f"# TOOLS ({normalized})\n\n- Define tool usage conventions.\n",
        "HEARTBEAT.md": f"# HEARTBEAT ({normalized})\n\n- Define health-check checklist.\n",
        "MEMORY.md": f"# MEMORY ({normalized})\n\n- Long-term memory for this profile.\n",
    }
    return templates.get(name, f"# {name} ({normalized})\n")


def resolve_profile_workspace(profile_id: str, workspace_raw: str) -> Path:
    normalized = normalize_profile_id(profile_id)
    raw = str(workspace_raw or "").strip()
    main_workspace = Path(default_workspace_for_profile("main")).resolve()
    default_workspace = Path(default_workspace_for_profile(normalized)).resolve()

    if not raw:
        return default_workspace

    candidate = Path(raw)
    resolved = candidate.resolve() if candidate.is_absolute() else candidate.resolve()

    if normalized != "main" and resolved == main_workspace:
        return default_workspace
    return resolved


def ensure_profile_workspace_scaffold(profile_id: str, workspace_raw: str) -> Path:
    normalized = normalize_profile_id(profile_id)
    workspace_dir = resolve_profile_workspace(normalized, workspace_raw)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "memory").mkdir(parents=True, exist_ok=True)

    for file_name in PROFILE_FILE_NAMES:
        target_path = workspace_dir / file_name
        if target_path.exists():
            continue
        target_path.write_text(default_profile_file_content(file_name, normalized), encoding="utf-8")
    return workspace_dir


class AuthCheckThread(QThread):
    completed = Signal(bool, str, object, str)

    def __init__(self, pending_profile: dict):
        super().__init__()
        self.pending_profile = pending_profile if isinstance(pending_profile, dict) else {}

    def run(self):
        try:
            source = str(self.pending_profile.get("modelSource") or "vertex").strip().lower()
            if source == SettingsDialog.PROVIDER_OLLAMA:
                ollama = self.pending_profile.get("ollama", {})
                if not isinstance(ollama, dict):
                    ollama = {}
                base_url = str(ollama.get("baseUrl") or "").strip()
                models = fetch_ollama_models(base_url)
                self.completed.emit(True, f"Ollama OK ({len(models)})", models, "")
                return

            if source == SettingsDialog.PROVIDER_API:
                api_provider = (
                    self.pending_profile.get("apiProvider", {})
                    if isinstance(self.pending_profile.get("apiProvider"), dict)
                    else {}
                )
                provider_id = normalize_provider_id(str(api_provider.get("providerId") or ""), "openai")
                base_url = str(api_provider.get("baseUrl") or "").strip() or "https://api.openai.com/v1"
                api_key = str(api_provider.get("apiKey") or "").strip()
                api_format = str(api_provider.get("apiFormat") or "openai-completions").strip().lower()
                if not api_key:
                    self.completed.emit(False, "실패", [], "API Key가 비어 있습니다.")
                    return
                if api_format == "anthropic-messages":
                    models = fetch_anthropic_models(base_url, api_key)
                else:
                    models = fetch_openai_compatible_models(base_url, api_key)
                self.completed.emit(True, f"API OK ({provider_id}:{len(models)})", models, "")
                return

            openclaw_dir = resolve_openclaw_dir()
            openclaw_mjs = openclaw_dir / "openclaw.mjs"
            if not openclaw_mjs.exists():
                raise RuntimeError("openclaw.mjs not found")

            env = os.environ.copy()
            vertex = self.pending_profile.get("vertex", {})
            if isinstance(vertex, dict):
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
            env["OPENCLAW_LOAD_SHELL_ENV"] = "1"

            code, stdout_text, stderr_text = run_process_capture(
                ["node", str(openclaw_mjs), "models", "status", "--json"],
                cwd=str(openclaw_dir),
                timeout=20,
                env=env,
            )
            if code != 0:
                detail = stderr_text.strip() or stdout_text.strip() or "models status failed"
                self.completed.emit(False, "실패", [], detail)
                return
            data = json.loads(stdout_text or "{}")
            missing = data.get("auth", {}).get("missingProvidersInUse", [])
            if "google-vertex" in missing:
                self.completed.emit(False, "실패", [], "")
                return
            self.completed.emit(True, "OK", [], "")
        except Exception as e:
            self.completed.emit(False, "실패", [], str(e))


class OllamaContextDetectThread(QThread):
    completed = Signal(bool, int, str)

    def __init__(self, base_url: str, model_name: str):
        super().__init__()
        self.base_url = str(base_url or "").strip()
        self.model_name = str(model_name or "").strip()

    def run(self):
        try:
            context_window, source = detect_ollama_context_window(self.base_url, self.model_name)
            if context_window and context_window > 0:
                self.completed.emit(True, int(context_window), source)
                return
            self.completed.emit(False, 0, source or "자동 감지 실패")
        except Exception as e:
            self.completed.emit(False, 0, str(e))


class OllamaToolsSupportThread(QThread):
    completed = Signal(object, str)

    def __init__(self, base_url: str, model_name: str, api_key: str):
        super().__init__()
        self.base_url = str(base_url or "").strip()
        self.model_name = str(model_name or "").strip()
        self.api_key = str(api_key or "").strip()

    def run(self):
        try:
            supported, detail = detect_ollama_tools_support(
                self.base_url,
                self.model_name,
                self.api_key,
            )
            self.completed.emit(supported, detail or "")
        except Exception as e:
            self.completed.emit(None, str(e))


class OllamaModelTaskThread(QThread):
    completed = Signal(str, bool, str, object)

    def __init__(
        self,
        action: str,
        base_url: str,
        folder: str = "",
        gguf_file: str = "",
        model_name: str = "",
        temperature: str = "0.7",
        remove_name: str = "",
    ):
        super().__init__()
        self.action = str(action or "").strip().lower()
        self.base_url = str(base_url or "").strip()
        self.folder = str(folder or "").strip()
        self.gguf_file = str(gguf_file or "").strip()
        self.model_name = str(model_name or "").strip()
        self.temperature = str(temperature or "0.7").strip()
        self.remove_name = str(remove_name or "").strip()

    def run(self):
        try:
            if self.action == "list":
                models = fetch_ollama_models_any(self.base_url)
                self.completed.emit("list", True, f"모델 {len(models)}개 로드 완료", models)
                return

            if self.action == "create":
                folder_path = Path(self.folder)
                if not folder_path.exists():
                    self.completed.emit("create", False, f"폴더가 없습니다: {folder_path}", {})
                    return
                gguf_path = folder_path / self.gguf_file
                if not gguf_path.exists():
                    self.completed.emit("create", False, f"GGUF 파일이 없습니다: {gguf_path}", {})
                    return
                if not self.model_name:
                    self.completed.emit("create", False, "등록 모델 이름을 입력하세요.", {})
                    return
                modelfile_path = folder_path / "Modelfile"
                modelfile_text = build_modelfile_text(self.gguf_file, self.temperature)
                modelfile_path.write_text(modelfile_text, encoding="utf-8")
                code, stdout_text, stderr_text = run_ollama_cli_capture(
                    ["create", self.model_name, "-f", str(modelfile_path)],
                    base_url=self.base_url,
                    cwd=str(folder_path),
                    timeout=1800,
                )
                if code != 0:
                    detail = (stderr_text or stdout_text or "").strip() or "ollama create failed"
                    self.completed.emit("create", False, detail, {})
                    return
                models = fetch_ollama_models_any(self.base_url)
                payload = {
                    "models": models,
                    "model": self.model_name,
                    "stdout": str(stdout_text or "").strip(),
                }
                self.completed.emit("create", True, f"등록 완료: {self.model_name}", payload)
                return

            if self.action == "remove":
                target = self.remove_name
                if not target:
                    self.completed.emit("remove", False, "제거할 모델을 선택하세요.", {})
                    return
                code, stdout_text, stderr_text = run_ollama_cli_capture(
                    ["rm", target],
                    base_url=self.base_url,
                    timeout=120,
                )
                if code != 0:
                    detail = (stderr_text or stdout_text or "").strip() or "ollama rm failed"
                    self.completed.emit("remove", False, detail, {})
                    return
                models = fetch_ollama_models_any(self.base_url)
                payload = {"models": models, "removed": target}
                self.completed.emit("remove", True, f"제거 완료: {target}", payload)
                return

            self.completed.emit(self.action, False, f"알 수 없는 작업: {self.action}", {})
        except Exception as e:
            self.completed.emit(self.action, False, str(e), {})


class GatewayActionThread(QThread):
    completed = Signal(str, bool, str)
    progress = Signal(str, int, int, str)

    def __init__(self, action: str, profile: dict):
        super().__init__()
        self.action = (action or "").strip().lower()
        self.profile = profile if isinstance(profile, dict) else {}
        self._last_progress_second: int = -1

    def _emit_progress(self, step: int, total: int, detail: str):
        self.progress.emit(self.action, step, total, detail)

    def _emit_wait_progress(self, step: int, total: int, prefix: str, elapsed: float, timeout_sec: float):
        elapsed_sec = int(elapsed)
        if elapsed_sec == self._last_progress_second:
            return
        self._last_progress_second = elapsed_sec
        remain_sec = max(0, int(timeout_sec - elapsed))
        self._emit_progress(step, total, f"{prefix} ({elapsed_sec}s 경과, {remain_sec}s 남음)")

    def start_gateway(self, force_restart: bool = False) -> tuple[bool, str]:
        total_steps = 5 if force_restart else 4
        self._last_progress_second = -1
        self._emit_progress(1, total_steps, "게이트웨이 상태 확인")
        baseline_pids: set[int] = set()
        baseline_running = check_gateway_running()
        if force_restart:
            baseline_pids = set(list_listening_pids(18789))
        if not force_restart and check_gateway_running():
            return True, "게이트웨이 이미 실행 중"
        script = resolve_gateway_script()
        self._emit_progress(2, total_steps, "시작 스크립트 확인")
        if not script.exists():
            return False, f"start_gateway.bat not found: {script}"
        env = build_gateway_env(self.profile)
        self._emit_progress(3, total_steps, "게이트웨이 시작 프로세스 실행")
        subprocess.Popen(
            ["cmd", "/c", str(script)],
            cwd=str(script.parent),
            env=env,
            **hidden_popen_kwargs(),
        )
        if force_restart:
            wait_timeout = 60.0
            started = time.time()
            saw_down = not baseline_running
            self._last_progress_second = -1
            while (time.time() - started) < wait_timeout:
                elapsed = time.time() - started
                running = check_gateway_running()
                current_pids = set(list_listening_pids(18789))
                if not running:
                    saw_down = True
                pid_changed = bool(baseline_pids) and bool(current_pids) and (current_pids != baseline_pids)
                elapsed_sec = int(elapsed)
                if elapsed_sec != self._last_progress_second:
                    self._last_progress_second = elapsed_sec
                    remain_sec = max(0, int(wait_timeout - elapsed))
                    if baseline_pids:
                        before_text = ",".join(str(pid) for pid in sorted(baseline_pids))
                        now_text = ",".join(str(pid) for pid in sorted(current_pids)) if current_pids else "탐지 중"
                        detail = (
                            f"재시작 전환 대기 ({elapsed_sec}s 경과, {remain_sec}s 남음, "
                            f"PID {before_text} -> {now_text})"
                        )
                    else:
                        detail = f"재시작 전환 대기 ({elapsed_sec}s 경과, {remain_sec}s 남음)"
                    self._emit_progress(4, total_steps, detail)
                if running and (pid_changed or saw_down or (not baseline_pids and elapsed >= 2.0)):
                    self._emit_progress(5, total_steps, "재시작 전환 확인 완료")
                    return True, "게이트웨이 재시작 완료"
                time.sleep(0.35)
            return False, "게이트웨이 재시작 확인 타임아웃"

        wait_timeout = 60.0
        if wait_gateway_state(
            True,
            timeout_sec=wait_timeout,
            on_tick=lambda elapsed: self._emit_wait_progress(
                4, total_steps, "포트 18789 연결 대기", elapsed, wait_timeout
            ),
        ):
            self._emit_progress(4, total_steps, "포트 연결 확인 완료")
            return True, "게이트웨이 시작 완료"
        return False, "게이트웨이 시작 타임아웃"

    def stop_gateway(self) -> tuple[bool, str]:
        total_steps = 4
        self._last_progress_second = -1
        self._emit_progress(1, total_steps, "게이트웨이 상태 확인")
        if not check_gateway_running():
            return True, "게이트웨이 이미 중지됨"
        self._emit_progress(2, total_steps, "게이트웨이 PID 탐지")
        pids = list_listening_pids(18789)
        if not pids:
            wait_timeout = 2.0
            if wait_gateway_state(
                False,
                timeout_sec=wait_timeout,
                on_tick=lambda elapsed: self._emit_wait_progress(
                    3, total_steps, "PID 미탐지 상태에서 종료 확인 대기", elapsed, wait_timeout
                ),
            ):
                return True, "게이트웨이 중지됨"
            return False, "게이트웨이 PID 탐지 실패"
        self._emit_progress(3, total_steps, f"종료 신호 전송 (PID {', '.join(str(pid) for pid in pids)})")
        killed, kill_error = kill_pids(pids)
        if not killed:
            return False, f"게이트웨이 종료 권한/프로세스 오류 ({kill_error})"
        wait_timeout = 10.0
        if wait_gateway_state(
            False,
            timeout_sec=wait_timeout,
            on_tick=lambda elapsed: self._emit_wait_progress(
                4, total_steps, "포트 종료 대기", elapsed, wait_timeout
            ),
        ):
            self._emit_progress(4, total_steps, "포트 종료 확인 완료")
            return True, "게이트웨이 중지 완료"
        return False, "게이트웨이 중지 실패"

    def run(self):
        try:
            if self.action == "start":
                ok, msg = self.start_gateway()
            elif self.action == "stop":
                ok, msg = self.stop_gateway()
            elif self.action == "restart":
                ok, msg = self.start_gateway(force_restart=True)
            else:
                ok, msg = False, f"unknown action: {self.action}"
            self.completed.emit(self.action, ok, msg)
        except Exception as e:
            self.completed.emit(self.action, False, str(e))


class ApplySettingsThread(QThread):
    completed = Signal(object)

    def __init__(self, profile_id: str, profile_settings: dict):
        super().__init__()
        self.profile_id = normalize_profile_id(profile_id)
        self.profile_settings = profile_settings if isinstance(profile_settings, dict) else {}

    def run_openclaw(self, openclaw_mjs: Path, args: list[str], timeout: int = 20) -> tuple[int, str, str]:
        return run_process_capture(
            ["node", str(openclaw_mjs), *args],
            cwd=str(openclaw_mjs.parent),
            timeout=timeout,
        )

    def read_agents_list(self, openclaw_mjs: Path) -> list[dict]:
        code, stdout_text, _ = self.run_openclaw(openclaw_mjs, ["config", "get", "--json", "agents.list"])
        if code != 0:
            return []
        try:
            parsed = json.loads(stdout_text or "[]")
            if isinstance(parsed, list):
                return [entry for entry in parsed if isinstance(entry, dict)]
        except Exception:
            return []
        return []

    def ensure_profile_agent_config(self, openclaw_mjs: Path) -> tuple[list[str], int]:
        errors: list[str] = []
        agent_id = normalize_agent_id(str(self.profile_settings.get("agentId") or self.profile_id))
        workspace = str(self.profile_settings.get("workspace") or "").strip()
        model = str(self.profile_settings.get("model") or "").strip() or "google-vertex/gemini-2.0-flash"
        if not workspace:
            workspace = str(default_workspace_for_profile(self.profile_id))

        agents = self.read_agents_list(openclaw_mjs)
        index = -1
        for idx, entry in enumerate(agents):
            if normalize_agent_id(str(entry.get("id") or "")) == agent_id:
                index = idx
                break
        if index < 0:
            index = len(agents)

        agent_payload = {
            "id": agent_id,
            "workspace": workspace,
            "model": {"primary": model},
        }
        code, stdout_text, stderr_text = self.run_openclaw(
            openclaw_mjs,
            [
                "config",
                "set",
                "--json",
                f"agents.list[{index}]",
                json.dumps(agent_payload, ensure_ascii=False),
            ],
        )
        if code != 0:
            # Fallback for older OpenClaw builds that may not accept object-level update.
            updates = [
                (f"agents.list[{index}].id", agent_id),
                (f"agents.list[{index}].workspace", workspace),
                (f"agents.list[{index}].model.primary", model),
            ]
            for path, value in updates:
                code_item, stdout_item, stderr_item = self.run_openclaw(
                    openclaw_mjs,
                    ["config", "set", path, value],
                )
                if code_item != 0:
                    errors.append(f"{path}: {stderr_item.strip() or stdout_item.strip()}")
        return errors, index

    def resolve_provider_id(self) -> str:
        source = str(self.profile_settings.get("modelSource") or "").strip().lower()
        model_ref = str(self.profile_settings.get("model") or "").strip().lower()
        if "/" in model_ref:
            return normalize_provider_id(model_ref.split("/", 1)[0], "openai")
        if source == SettingsDialog.PROVIDER_OLLAMA:
            return "ollama"
        if source == SettingsDialog.PROVIDER_API:
            api_provider = (
                self.profile_settings.get("apiProvider", {})
                if isinstance(self.profile_settings.get("apiProvider"), dict)
                else {}
            )
            return normalize_provider_id(str(api_provider.get("providerId") or ""), "openai")
        return "google-vertex"

    def ensure_profile_tool_mode_policy(self, openclaw_mjs: Path, agent_index: int | None = None) -> list[str]:
        errors: list[str] = []
        if not isinstance(agent_index, int) or agent_index < 0:
            agent_id = normalize_agent_id(str(self.profile_settings.get("agentId") or self.profile_id))
            agents = self.read_agents_list(openclaw_mjs)
            agent_index = -1
            for idx, entry in enumerate(agents):
                if normalize_agent_id(str(entry.get("id") or "")) == agent_id:
                    agent_index = idx
                    break
        if agent_index < 0:
            return ["tool policy: agent not found"]
        tool_mode = str(self.profile_settings.get("toolMode") or "auto").strip().lower()
        if tool_mode not in {"auto", "chat", "agent"}:
            tool_mode = "auto"
        provider_id = self.resolve_provider_id()
        source = str(self.profile_settings.get("modelSource") or "").strip().lower()
        ollama = self.profile_settings.get("ollama", {})
        supports_tools = (
            ollama.get("supportsTools")
            if isinstance(ollama, dict) and isinstance(ollama.get("supportsTools"), bool)
            else None
        )
        if tool_mode == "chat":
            allow_payload = "[]"
        elif tool_mode == "agent":
            allow_payload = "[\"*\"]"
        else:
            allow_payload = "[\"*\"]" if supports_tools is True else "[]"
        code, stdout_text, stderr_text = self.run_openclaw(
            openclaw_mjs,
            [
                "config",
                "set",
                "--json",
                f"agents.list[{agent_index}].tools.byProvider.{provider_id}.allow",
                allow_payload,
            ],
        )
        if code != 0:
            errors.append(
                f"agents.list[{agent_index}].tools.byProvider.{provider_id}.allow: "
                + (stderr_text.strip() or stdout_text.strip() or "set failed")
            )
        return errors

    def run(self):
        errors: list[str] = []
        try:
            workspace = str(self.profile_settings.get("workspace") or "")
            prepared_workspace = ensure_profile_workspace_scaffold(self.profile_id, workspace)
            self.profile_settings["workspace"] = str(prepared_workspace)

            openclaw_dir = resolve_openclaw_dir()
            openclaw_mjs = openclaw_dir / "openclaw.mjs"
            if openclaw_mjs.exists():
                source = str(self.profile_settings.get("modelSource") or "").strip().lower()
                if source == SettingsDialog.PROVIDER_OLLAMA:
                    ollama = (
                        self.profile_settings.get("ollama", {})
                        if isinstance(self.profile_settings.get("ollama"), dict)
                        else {}
                    )
                    base_url = str(ollama.get("baseUrl") or "http://127.0.0.1:11434/v1").strip()
                    api_key = str(ollama.get("apiKey") or "ollama-local").strip() or "ollama-local"
                    preferred_model = str(ollama.get("modelName") or "").strip()
                    selected_model = str(self.profile_settings.get("model") or "").strip()
                    if selected_model.startswith("ollama/"):
                        selected_model = selected_model.split("/", 1)[1].strip()
                    model_ids: list[str] = []
                    for item in (
                        preferred_model,
                        selected_model,
                    ):
                        model_id = str(item or "").strip()
                        if model_id and model_id not in model_ids:
                            model_ids.append(model_id)
                    if not model_ids:
                        model_ids = ["llama3.3"]
                    context_window = clamp_int(
                        ollama.get("contextWindow"),
                        DEFAULT_OLLAMA_CONTEXT_WINDOW,
                        OPENCLAW_MIN_CONTEXT_WINDOW,
                        MAX_OLLAMA_CONTEXT_WINDOW,
                    )
                    max_tokens = max(1024, min(8192, context_window // 2))
                    provider_payload = {
                        "baseUrl": base_url,
                        "apiKey": api_key,
                        "api": "openai-completions",
                        "models": [
                            {
                                "id": model_id,
                                "name": model_id,
                                "reasoning": False,
                                "input": ["text"],
                                "cost": {
                                    "input": 0,
                                    "output": 0,
                                    "cacheRead": 0,
                                    "cacheWrite": 0,
                                },
                                "contextWindow": context_window,
                                "maxTokens": max_tokens,
                            }
                            for model_id in model_ids
                        ],
                    }
                    code, stdout_text, stderr_text = self.run_openclaw(
                        openclaw_mjs,
                        [
                            "config",
                            "set",
                            "--json",
                            "models.providers.ollama",
                            json.dumps(provider_payload, ensure_ascii=False),
                        ],
                    )
                    if code != 0:
                        errors.append(
                            "models.providers.ollama: "
                            + (stderr_text.strip() or stdout_text.strip() or "set failed")
                        )
                elif source == SettingsDialog.PROVIDER_API:
                    api_provider = (
                        self.profile_settings.get("apiProvider", {})
                        if isinstance(self.profile_settings.get("apiProvider"), dict)
                        else {}
                    )
                    provider_id = normalize_provider_id(str(api_provider.get("providerId") or ""), "openai")
                    base_url = str(api_provider.get("baseUrl") or "").strip() or "https://api.openai.com/v1"
                    api_key = str(api_provider.get("apiKey") or "").strip()
                    api_format = str(api_provider.get("apiFormat") or "openai-completions").strip().lower()
                    if api_format not in {"openai-completions", "openai-responses", "anthropic-messages"}:
                        api_format = "openai-completions"
                    preferred_model = str(api_provider.get("modelName") or "").strip()
                    selected_model = str(self.profile_settings.get("model") or "").strip()
                    selected_provider = ""
                    if "/" in selected_model:
                        selected_provider, selected_suffix = selected_model.split("/", 1)
                        selected_provider = normalize_provider_id(selected_provider, provider_id)
                        selected_model = selected_suffix.strip()
                    model_ids: list[str] = []
                    for item in (
                        preferred_model,
                        selected_model if (not selected_provider or selected_provider == provider_id) else "",
                    ):
                        model_id = str(item or "").strip()
                        if model_id and model_id not in model_ids:
                            model_ids.append(model_id)
                    if not model_ids:
                        model_ids = ["gpt-4.1-mini"]
                    context_window = clamp_int(
                        api_provider.get("contextWindow"),
                        DEFAULT_API_CONTEXT_WINDOW,
                        OPENCLAW_MIN_CONTEXT_WINDOW,
                        MAX_API_CONTEXT_WINDOW,
                    )
                    max_tokens = max(1024, min(8192, context_window // 2))
                    provider_payload = {
                        "baseUrl": base_url,
                        "apiKey": api_key,
                        "api": api_format,
                        "models": [
                            {
                                "id": model_id,
                                "name": model_id,
                                "reasoning": False,
                                "input": ["text"],
                                "cost": {
                                    "input": 0,
                                    "output": 0,
                                    "cacheRead": 0,
                                    "cacheWrite": 0,
                                },
                                "contextWindow": context_window,
                                "maxTokens": max_tokens,
                            }
                            for model_id in model_ids
                        ],
                    }
                    code, stdout_text, stderr_text = self.run_openclaw(
                        openclaw_mjs,
                        [
                            "config",
                            "set",
                            "--json",
                            f"models.providers.{provider_id}",
                            json.dumps(provider_payload, ensure_ascii=False),
                        ],
                    )
                    if code != 0:
                        errors.append(
                            f"models.providers.{provider_id}: "
                            + (stderr_text.strip() or stdout_text.strip() or "set failed")
                        )

                agent_errors, agent_index = self.ensure_profile_agent_config(openclaw_mjs)
                errors.extend(agent_errors)
                errors.extend(self.ensure_profile_tool_mode_policy(openclaw_mjs, agent_index))
        except Exception as e:
            errors.append(str(e))
        self.completed.emit(errors)


class SettingsDialog(QDialog):
    PROVIDER_VERTEX = "vertex"
    PROVIDER_OLLAMA = "ollama"
    PROVIDER_API = "api"
    API_PROVIDER_PRESETS = {
        "openai": {
            "label": "OpenAI",
            "providerId": "openai",
            "baseUrl": "https://api.openai.com/v1",
            "apiFormat": "openai-completions",
            "modelName": "gpt-4.1-mini",
        },
        "gemini": {
            "label": "Google Gemini API",
            "providerId": "gemini-api",
            "baseUrl": "https://generativelanguage.googleapis.com/v1beta/openai",
            "apiFormat": "openai-completions",
            "modelName": "gemini-2.0-flash",
        },
        "anthropic": {
            "label": "Anthropic API",
            "providerId": "anthropic-api",
            "baseUrl": "https://api.anthropic.com/v1",
            "apiFormat": "anthropic-messages",
            "modelName": "claude-sonnet-4-5",
        },
        "openrouter": {
            "label": "OpenRouter",
            "providerId": "openrouter",
            "baseUrl": "https://openrouter.ai/api/v1",
            "apiFormat": "openai-completions",
            "modelName": "openai/gpt-4.1-mini",
        },
        "groq": {
            "label": "Groq",
            "providerId": "groq",
            "baseUrl": "https://api.groq.com/openai/v1",
            "apiFormat": "openai-completions",
            "modelName": "llama-3.3-70b-versatile",
        },
        "xai": {
            "label": "xAI",
            "providerId": "xai",
            "baseUrl": "https://api.x.ai/v1",
            "apiFormat": "openai-completions",
            "modelName": "grok-4",
        },
        "lmstudio": {
            "label": "LM Studio (local)",
            "providerId": "lmstudio",
            "baseUrl": "http://127.0.0.1:1234/v1",
            "apiFormat": "openai-completions",
            "modelName": "local-model",
        },
        "vllm": {
            "label": "vLLM (local)",
            "providerId": "vllm",
            "baseUrl": "http://127.0.0.1:8000/v1",
            "apiFormat": "openai-completions",
            "modelName": "local-model",
        },
    }

    def __init__(self, parent=None, initial_profile: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("GhostRelay 설정")
        self.setModal(True)
        self.setMinimumSize(760, 680)
        self.resize(860, 760)
        self.setSizeGripEnabled(True)
        self.settings = load_settings()
        if initial_profile:
            requested = normalize_profile_id(initial_profile)
            available = set(list_profiles(self.settings))
            self.active_profile_id = requested if requested in available else get_active_profile_id(self.settings)
        else:
            self.active_profile_id = get_active_profile_id(self.settings)
        self.profile_settings = get_profile(self.settings, self.active_profile_id)
        self._profile_loading = False
        self.auth_check_thread: Optional[AuthCheckThread] = None
        self.ollama_context_thread: Optional[OllamaContextDetectThread] = None
        self.ollama_tools_thread: Optional[OllamaToolsSupportThread] = None
        self.apply_thread: Optional[ApplySettingsThread] = None
        self._auth_popup_pending = False
        self._context_popup_pending = False
        self._tools_popup_pending = False
        self.detected_context_window: Optional[int] = None
        self.detected_context_source: str = ""
        self.detected_tools_supported: Optional[bool] = None
        self.detected_tools_detail: str = ""
        self.setStyleSheet("""
            QDialog {
                background: #0f141b;
                color: #e5e7eb;
            }
            QLabel {
                color: #e5e7eb;
            }
            QLineEdit, QComboBox {
                background: #0b0f14;
                color: #e5e7eb;
                border: 1px solid #2a2f3a;
                border-radius: 6px;
                padding: 4px 6px;
                min-height: 30px;
            }
            QPushButton {
                background: #1f2937;
                color: #e5e7eb;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 4px 8px;
                min-height: 30px;
            }
            QPushButton:hover {
                background: #374151;
            }
        """)
        self.init_ui()

    def current_source(self) -> str:
        source = str(self.source_combo.currentData() or "").strip().lower()
        if source not in {self.PROVIDER_VERTEX, self.PROVIDER_OLLAMA, self.PROVIDER_API}:
            return self.PROVIDER_VERTEX
        return source

    def current_tool_mode(self) -> str:
        mode = str(self.tool_mode_combo.currentData() or "").strip().lower()
        if mode not in {"auto", "chat", "agent"}:
            return "auto"
        return mode

    def normalize_ollama_model(self, model_name: str) -> str:
        raw = (model_name or "").strip()
        if not raw:
            raw = "llama3.3"
        if raw.startswith("ollama/"):
            return raw
        return f"ollama/{raw}"

    def get_api_provider_id(self) -> str:
        return normalize_provider_id(self.api_provider_id_input.text().strip(), "openai")

    def normalize_api_model(self, provider_id: str, model_name: str) -> str:
        provider = normalize_provider_id(provider_id, "openai")
        raw_model = str(model_name or "").strip()
        if not raw_model:
            raw_model = "gpt-4.1-mini"
        if raw_model.startswith(f"{provider}/"):
            return raw_model
        return f"{provider}/{raw_model}"

    def infer_api_preset_key(self, provider_id: str, base_url: str) -> str:
        normalized_provider = normalize_provider_id(provider_id, "openai")
        normalized_base = str(base_url or "").strip().rstrip("/")
        for preset_key, preset in self.API_PROVIDER_PRESETS.items():
            preset_provider = normalize_provider_id(str(preset.get("providerId") or ""), "openai")
            preset_base = str(preset.get("baseUrl") or "").strip().rstrip("/")
            if normalized_provider == preset_provider and normalized_base == preset_base:
                return preset_key
        return "custom"

    def apply_api_preset(self, preset_key: str, trigger_auth: bool = True):
        key = str(preset_key or "").strip().lower()
        if key == "custom":
            return
        preset = self.API_PROVIDER_PRESETS.get(key)
        if not isinstance(preset, dict):
            return
        self.api_provider_id_input.setText(str(preset.get("providerId") or "openai"))
        self.api_base_url_input.setText(str(preset.get("baseUrl") or "https://api.openai.com/v1"))
        self.api_model_input.setText(str(preset.get("modelName") or "gpt-4.1-mini"))
        api_format = str(preset.get("apiFormat") or "openai-completions").strip().lower()
        format_index = self.api_format_combo.findData(api_format)
        self.api_format_combo.setCurrentIndex(format_index if format_index >= 0 else 0)
        self.sync_api_model_preview()
        if trigger_auth:
            self.check_auth(show_popup=False)

    def on_api_preset_changed(self):
        if self._profile_loading:
            return
        preset_key = str(self.api_preset_combo.currentData() or "custom").strip().lower()
        self.apply_api_preset(preset_key, trigger_auth=False)

    def sync_api_preset_from_inputs(self):
        if self._profile_loading:
            return
        preset_key = self.infer_api_preset_key(
            self.api_provider_id_input.text().strip(),
            self.api_base_url_input.text().strip(),
        )
        current_key = str(self.api_preset_combo.currentData() or "").strip().lower()
        if current_key == preset_key:
            return
        self.api_preset_combo.blockSignals(True)
        preset_index = self.api_preset_combo.findData(preset_key)
        if preset_index < 0:
            preset_index = self.api_preset_combo.findData("custom")
        if preset_index < 0:
            preset_index = 0
        self.api_preset_combo.setCurrentIndex(preset_index)
        self.api_preset_combo.blockSignals(False)

    def build_pending_profile(self, profile_id: str | None = None) -> dict:
        pid = normalize_profile_id(profile_id or self.active_profile_id)
        source = self.current_source()
        ollama_model_name = self.ollama_model_input.text().strip() or "llama3.3"
        api_provider_id = self.get_api_provider_id()
        api_model_name = self.api_model_input.text().strip() or "gpt-4.1-mini"
        if source == self.PROVIDER_OLLAMA:
            effective_model = self.normalize_ollama_model(ollama_model_name)
        elif source == self.PROVIDER_API:
            effective_model = self.normalize_api_model(api_provider_id, api_model_name)
        else:
            effective_model = self.model_combo.currentText().strip() or "google-vertex/gemini-2.0-flash"
        workspace = str(resolve_profile_workspace(pid, self.workspace_input.text().strip()))
        agent_id_value = normalize_agent_id(self.agent_id_input.text().strip() or pid)
        if pid != "main" and agent_id_value == "main":
            agent_id_value = pid
        return {
            "model": effective_model,
            "modelSource": source,
            "toolMode": self.current_tool_mode(),
            "sessionKey": self.session_key_input.text().strip(),
            "agentId": agent_id_value,
            "workspace": workspace,
            "vertex": {
                "serviceAccountPath": self.sa_input.text().strip(),
                "project": self.project_input.text().strip(),
                "location": self.location_input.text().strip(),
            },
            "ollama": {
                "baseUrl": self.ollama_base_url_input.text().strip() or "http://127.0.0.1:11434/v1",
                "apiKey": self.ollama_api_key_input.text().strip() or "ollama-local",
                "modelName": ollama_model_name,
                "supportsTools": self.detected_tools_supported,
                "contextWindow": clamp_int(
                    self.ollama_context_input.text(),
                    DEFAULT_OLLAMA_CONTEXT_WINDOW,
                    OPENCLAW_MIN_CONTEXT_WINDOW,
                    MAX_OLLAMA_CONTEXT_WINDOW,
                ),
                "injectionLevel": str(self.ollama_injection_combo.currentData() or "medium"),
                "historyTurns": clamp_int(
                    self.ollama_history_slider.value(),
                    6,
                    1,
                    20,
                ),
            },
            "apiProvider": {
                "providerId": api_provider_id,
                "baseUrl": self.api_base_url_input.text().strip() or "https://api.openai.com/v1",
                "apiKey": self.api_key_input.text().strip(),
                "modelName": api_model_name,
                "apiFormat": str(self.api_format_combo.currentData() or "openai-completions"),
                "contextWindow": clamp_int(
                    self.api_context_input.text(),
                    DEFAULT_API_CONTEXT_WINDOW,
                    OPENCLAW_MIN_CONTEXT_WINDOW,
                    MAX_API_CONTEXT_WINDOW,
                ),
            },
        }

    def build_storage_settings(self, profile_id: str | None = None) -> dict:
        pid = normalize_profile_id(profile_id or self.active_profile_id)
        settings = load_settings()
        profiles = settings.get("profiles", {}) if isinstance(settings.get("profiles"), dict) else {}
        normalized_profiles: dict[str, dict] = {}
        for existing_id in list_profiles(settings):
            normalized_profiles[existing_id] = get_profile(settings, existing_id)
        normalized_profiles[pid] = self.build_pending_profile(pid)
        settings["profiles"] = normalized_profiles
        settings["activeProfile"] = pid
        settings["alwaysOnTop"] = bool(settings.get("alwaysOnTop", True))
        ui = settings.get("ui", {}) if isinstance(settings.get("ui"), dict) else {}
        ui["language"] = str(self.lang_combo.currentData() or "ko")
        settings["ui"] = ui
        return settings

    def refresh_profile_selector(self, selected_id: str | None = None):
        target_id = normalize_profile_id(selected_id or self.active_profile_id)
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        settings = load_settings()
        for profile_id in list_profiles(settings):
            self.profile_combo.addItem(profile_id, profile_id)
        idx = self.profile_combo.findData(target_id)
        if idx < 0 and self.profile_combo.count() > 0:
            idx = 0
        if idx >= 0:
            self.profile_combo.setCurrentIndex(idx)
        self.profile_combo.blockSignals(False)

    def load_profile_into_form(self, profile_id: str):
        pid = normalize_profile_id(profile_id)
        self.settings = load_settings()
        self.active_profile_id = pid
        profile = get_profile(self.settings, pid)
        self.profile_settings = profile
        self._profile_loading = True
        try:
            source = str(profile.get("modelSource") or self.PROVIDER_VERTEX).strip().lower()
            if source not in {self.PROVIDER_VERTEX, self.PROVIDER_OLLAMA, self.PROVIDER_API}:
                source = self.PROVIDER_VERTEX
            source_idx = self.source_combo.findData(source)
            self.source_combo.setCurrentIndex(source_idx if source_idx >= 0 else 0)

            model = str(profile.get("model") or "google-vertex/gemini-2.0-flash").strip()
            if self.model_combo.findText(model) == -1:
                self.model_combo.addItem(model)
            self.model_combo.setCurrentText(model)

            self.agent_id_input.setText(str(profile.get("agentId") or pid))
            self.workspace_input.setText(str(resolve_profile_workspace(pid, str(profile.get("workspace") or ""))))
            self.session_key_input.setText(str(profile.get("sessionKey") or ""))
            mode = str(profile.get("toolMode") or "auto").strip().lower()
            mode_index = self.tool_mode_combo.findData(mode if mode in {"auto", "chat", "agent"} else "auto")
            self.tool_mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)

            vertex = profile.get("vertex", {}) if isinstance(profile.get("vertex"), dict) else {}
            self.sa_input.setText(str(vertex.get("serviceAccountPath") or ""))
            self.project_input.setText(str(vertex.get("project") or ""))
            self.location_input.setText(str(vertex.get("location") or ""))

            ollama = profile.get("ollama", {}) if isinstance(profile.get("ollama"), dict) else {}
            self.ollama_base_url_input.setText(str(ollama.get("baseUrl") or "http://127.0.0.1:11434/v1"))
            model_name = str(ollama.get("modelName") or "").strip()
            if not model_name and model.startswith("ollama/"):
                model_name = model.split("/", 1)[1]
            self.ollama_model_input.setText(model_name or "llama3.3")
            self.ollama_api_key_input.setText(str(ollama.get("apiKey") or "ollama-local"))
            self.ollama_context_input.setText(
                str(
                    clamp_int(
                        ollama.get("contextWindow"),
                        DEFAULT_OLLAMA_CONTEXT_WINDOW,
                        OPENCLAW_MIN_CONTEXT_WINDOW,
                        MAX_OLLAMA_CONTEXT_WINDOW,
                    )
                )
            )
            injection_level = str(ollama.get("injectionLevel") or "medium").strip().lower()
            injection_index = self.ollama_injection_combo.findData(
                injection_level if injection_level in {"weak", "medium", "strong"} else "medium"
            )
            self.ollama_injection_combo.setCurrentIndex(injection_index if injection_index >= 0 else 1)
            history_turns = clamp_int(ollama.get("historyTurns"), 6, 1, 20)
            self.ollama_history_slider.setValue(history_turns)
            self.on_ollama_history_turns_changed(history_turns)
            self.detected_context_window = None
            self.detected_context_source = ""
            supports_tools = ollama.get("supportsTools")
            self.detected_tools_supported = supports_tools if isinstance(supports_tools, bool) else None
            self.detected_tools_detail = ""
            self.sync_ollama_model_preview()

            api_provider = (
                profile.get("apiProvider", {})
                if isinstance(profile.get("apiProvider"), dict)
                else {}
            )
            provider_id = normalize_provider_id(str(api_provider.get("providerId") or ""), "openai")
            self.api_provider_id_input.setText(provider_id)
            self.api_base_url_input.setText(str(api_provider.get("baseUrl") or "https://api.openai.com/v1"))
            self.api_key_input.setText(str(api_provider.get("apiKey") or ""))
            api_model_name = str(api_provider.get("modelName") or "").strip()
            if not api_model_name:
                if "/" in model:
                    model_provider, model_suffix = model.split("/", 1)
                    if normalize_provider_id(model_provider, provider_id) == provider_id:
                        api_model_name = model_suffix.strip()
            self.api_model_input.setText(api_model_name or "gpt-4.1-mini")
            api_format = str(api_provider.get("apiFormat") or "openai-completions").strip().lower()
            if api_format not in {"openai-completions", "openai-responses", "anthropic-messages"}:
                api_format = "openai-completions"
            format_index = self.api_format_combo.findData(api_format)
            self.api_format_combo.setCurrentIndex(format_index if format_index >= 0 else 0)
            self.api_context_input.setText(
                str(
                    clamp_int(
                        api_provider.get("contextWindow"),
                        DEFAULT_API_CONTEXT_WINDOW,
                        OPENCLAW_MIN_CONTEXT_WINDOW,
                        MAX_API_CONTEXT_WINDOW,
                    )
                )
            )
            preset_key = self.infer_api_preset_key(
                self.api_provider_id_input.text().strip(),
                self.api_base_url_input.text().strip(),
            )
            self.api_preset_combo.blockSignals(True)
            preset_index = self.api_preset_combo.findData(preset_key)
            self.api_preset_combo.setCurrentIndex(preset_index if preset_index >= 0 else 0)
            self.api_preset_combo.blockSignals(False)
            self.set_api_models([])
            self.sync_api_model_preview()
        finally:
            self._profile_loading = False
        self.sync_source_visibility()
        self.update_tools_support_hint()
        self.check_auth()

    def on_profile_changed(self):
        if self._profile_loading:
            return
        profile_id = str(self.profile_combo.currentData() or "").strip()
        if not profile_id:
            return
        self.load_profile_into_form(profile_id)

    def create_profile(self):
        text, ok = QInputDialog.getText(self, "새 프로필", "프로필 이름(영문/숫자/-/_):")
        if not ok:
            return
        profile_id = normalize_profile_id(text)
        if not profile_id:
            return
        settings = load_settings()
        profiles = settings.get("profiles", {}) if isinstance(settings.get("profiles"), dict) else {}
        if profile_id in profiles:
            self.refresh_profile_selector(profile_id)
            self.load_profile_into_form(profile_id)
            return
        created_profile = default_profile(profile_id)
        prepared_workspace = ensure_profile_workspace_scaffold(
            profile_id,
            str(created_profile.get("workspace") or ""),
        )
        created_profile["workspace"] = str(prepared_workspace)
        profiles[profile_id] = created_profile
        settings["profiles"] = profiles
        settings["activeProfile"] = profile_id
        save_settings(settings)
        self.refresh_profile_selector(profile_id)
        self.load_profile_into_form(profile_id)

    def delete_current_profile(self):
        profile_id = str(self.profile_combo.currentData() or "").strip()
        if normalize_profile_id(profile_id) == "main":
            QMessageBox.warning(self, "삭제 불가", "main 프로필은 삭제할 수 없습니다.")
            return
        settings = load_settings()
        profiles = settings.get("profiles", {}) if isinstance(settings.get("profiles"), dict) else {}
        if profile_id in profiles:
            profiles.pop(profile_id, None)
        if not profiles:
            profiles["main"] = default_profile("main")
        settings["profiles"] = profiles
        settings["activeProfile"] = "main" if "main" in profiles else next(iter(profiles.keys()))
        save_settings(settings)
        self.refresh_profile_selector(str(settings["activeProfile"]))
        self.load_profile_into_form(str(settings["activeProfile"]))

    def set_auth_status(self, ok: Optional[bool], label: str):
        self.auth_status.setText(f"인증 상태: {label} ●")
        if ok is None:
            self.auth_status.setStyleSheet("color: #9ca3af; font-size: 11px;")
            return
        if ok:
            self.auth_status.setStyleSheet("color: #22c55e; font-size: 11px;")
            return
        self.auth_status.setStyleSheet("color: #ef4444; font-size: 11px;")

    def set_ollama_models(self, models: list[str]):
        current = self.ollama_model_input.text().strip()
        self.ollama_models_combo.blockSignals(True)
        self.ollama_models_combo.clear()
        selected_model = ""
        if models:
            self.ollama_models_combo.addItems(models)
            self.ollama_models_combo.setEnabled(True)
            if current in models:
                self.ollama_models_combo.setCurrentText(current)
                selected_model = current
            else:
                self.ollama_models_combo.setCurrentIndex(0)
                selected_model = self.ollama_models_combo.currentText().strip()
        else:
            self.ollama_models_combo.addItem("(모델 없음)")
            self.ollama_models_combo.setEnabled(False)
        self.ollama_models_combo.blockSignals(False)
        if selected_model:
            self.on_ollama_model_selected(selected_model)

    def load_ollama_models(self, show_popup: bool = False):
        self.check_auth(show_popup=show_popup)

    def on_ollama_model_selected(self, model_name: str):
        selected = str(model_name or "").strip()
        if not selected or selected == "(모델 없음)":
            return
        self.ollama_model_input.setText(selected)
        self.sync_ollama_model_preview()
        self.detect_ollama_context(show_popup=False)
        self.detect_ollama_tools_support(show_popup=False)

    def set_api_models(self, models: list[str]):
        current = self.api_model_input.text().strip()
        self.api_models_combo.blockSignals(True)
        self.api_models_combo.clear()
        selected_model = ""
        if models:
            self.api_models_combo.addItems(models)
            self.api_models_combo.setEnabled(True)
            if current in models:
                self.api_models_combo.setCurrentText(current)
                selected_model = current
            else:
                self.api_models_combo.setCurrentIndex(0)
                selected_model = self.api_models_combo.currentText().strip()
        else:
            self.api_models_combo.addItem("(모델 없음)")
            self.api_models_combo.setEnabled(False)
        self.api_models_combo.blockSignals(False)
        if selected_model:
            self.on_api_model_selected(selected_model)

    def load_api_models(self, show_popup: bool = False):
        self.check_auth(show_popup=show_popup)

    def on_api_model_selected(self, model_name: str):
        selected = str(model_name or "").strip()
        if not selected or selected == "(모델 없음)":
            return
        self.api_model_input.setText(selected)
        self.sync_api_model_preview()

    def sync_api_model_preview(self):
        if self.current_source() != self.PROVIDER_API:
            return
        provider_id = self.get_api_provider_id()
        model_name = self.api_model_input.text().strip()
        self.model_combo.setCurrentText(self.normalize_api_model(provider_id, model_name))

    def update_tools_support_hint(self):
        if not hasattr(self, "tool_support_hint"):
            return
        source = self.current_source()
        if source == self.PROVIDER_API:
            self.tool_support_hint.setText(
                "API 모드: 모델별 tools 지원 정책이 다릅니다. 필요 시 대화 전용(도구 OFF)으로 사용하세요."
            )
            self.tool_support_hint.setStyleSheet("color: #9ca3af; font-size: 11px;")
            return
        if source != self.PROVIDER_OLLAMA:
            self.tool_support_hint.setText("Vertex 모드: 도구 가능 여부는 OpenClaw/모델 정책에 따릅니다.")
            self.tool_support_hint.setStyleSheet("color: #9ca3af; font-size: 11px;")
            return
        supported = self.detected_tools_supported
        detail = str(self.detected_tools_detail or "").strip()
        if supported is True:
            self.tool_support_hint.setText("✅ 현재 Ollama 모델은 tool-calling을 지원합니다.")
            self.tool_support_hint.setStyleSheet("color: #22c55e; font-size: 11px;")
            return
        if supported is False:
            message = "❌ 현재 Ollama 모델은 tool-calling 미지원입니다. 대화 전용 모드를 사용하세요."
            if detail:
                message += f" ({detail})"
            self.tool_support_hint.setText(message)
            self.tool_support_hint.setStyleSheet("color: #ef4444; font-size: 11px;")
            return
        base = "ℹ Ollama tool-calling 지원 여부 미확인"
        if detail:
            base += f" ({detail})"
        self.tool_support_hint.setText(base)
        self.tool_support_hint.setStyleSheet("color: #9ca3af; font-size: 11px;")

    def on_tool_mode_changed(self):
        if self._profile_loading:
            return
        self.enforce_tool_mode_compatibility(show_popup=False)

    def enforce_tool_mode_compatibility(self, show_popup: bool = False) -> bool:
        if self.current_source() != self.PROVIDER_OLLAMA:
            return True
        mode = self.current_tool_mode()
        supported = self.detected_tools_supported
        if mode == "chat":
            return True
        if mode == "auto" and supported is False:
            idx = self.tool_mode_combo.findData("chat")
            if idx >= 0:
                self.tool_mode_combo.blockSignals(True)
                self.tool_mode_combo.setCurrentIndex(idx)
                self.tool_mode_combo.blockSignals(False)
            if show_popup:
                QMessageBox.information(
                    self,
                    "도구 모드 자동 전환",
                    "선택한 Ollama 모델이 tools를 지원하지 않아 대화 전용(도구 OFF)으로 전환했습니다.",
                )
            return True
        if mode == "agent" and supported is False:
            if show_popup:
                QMessageBox.warning(
                    self,
                    "모드 호환성 오류",
                    "선택한 Ollama 모델은 tools를 지원하지 않습니다.\n"
                    "대화 전용(도구 OFF) 또는 tools 지원 모델로 변경하세요.",
                )
            return False
        return True

    def detect_ollama_tools_support(self, show_popup: bool = False):
        if self.current_source() != self.PROVIDER_OLLAMA:
            self.detected_tools_supported = None
            self.detected_tools_detail = ""
            self.update_tools_support_hint()
            return
        if self.ollama_tools_thread and self.ollama_tools_thread.isRunning():
            return
        model_name = str(self.ollama_model_input.text() or "").strip()
        if not model_name:
            self.detected_tools_supported = None
            self.detected_tools_detail = "모델 이름이 비어 있습니다."
            self.update_tools_support_hint()
            return
        base_url = str(self.ollama_base_url_input.text() or "").strip() or "http://127.0.0.1:11434/v1"
        api_key = str(self.ollama_api_key_input.text() or "").strip() or "ollama-local"
        self._tools_popup_pending = bool(show_popup)
        self.detected_tools_supported = None
        self.detected_tools_detail = "검사 중..."
        self.update_tools_support_hint()
        self.ollama_tools_thread = OllamaToolsSupportThread(base_url, model_name, api_key)
        self.ollama_tools_thread.completed.connect(self.on_ollama_tools_support_completed)
        self.ollama_tools_thread.start()

    def on_ollama_tools_support_completed(self, supported_obj: object, detail: str):
        supported: Optional[bool]
        if isinstance(supported_obj, bool):
            supported = supported_obj
        else:
            supported = None
        self.detected_tools_supported = supported
        self.detected_tools_detail = str(detail or "").strip()
        self.update_tools_support_hint()
        self.enforce_tool_mode_compatibility(show_popup=False)
        if self._tools_popup_pending:
            if supported is True:
                QMessageBox.information(self, "도구 지원 확인", "현재 Ollama 모델은 tool-calling을 지원합니다.")
            elif supported is False:
                QMessageBox.warning(
                    self,
                    "도구 미지원 모델",
                    "현재 Ollama 모델은 tool-calling을 지원하지 않습니다.\n대화 전용(도구 OFF)으로 사용하세요.",
                )
            elif detail:
                QMessageBox.information(self, "도구 지원 미확인", detail)
        self._tools_popup_pending = False
        self.ollama_tools_thread = None

    def set_ollama_context_hint(self, text: str, color: str = "#9ca3af"):
        if not hasattr(self, "ollama_context_hint"):
            return
        self.ollama_context_hint.setText(text)
        self.ollama_context_hint.setStyleSheet(f"color: {color}; font-size: 11px;")

    def get_ollama_context_range(self) -> tuple[int, Optional[int], int]:
        range_min = OPENCLAW_MIN_CONTEXT_WINDOW
        range_max: Optional[int] = None
        if self.detected_context_window:
            range_max = max(
                range_min,
                min(int(self.detected_context_window), MAX_OLLAMA_CONTEXT_WINDOW),
            )
        recommended = min(DEFAULT_OLLAMA_CONTEXT_WINDOW, range_max) if range_max else DEFAULT_OLLAMA_CONTEXT_WINDOW
        recommended = max(range_min, recommended)
        return range_min, range_max, recommended

    def validate_ollama_context(self, show_popup: bool = False) -> bool:
        if self.current_source() != self.PROVIDER_OLLAMA:
            self.set_ollama_context_hint("")
            return True
        range_min, range_max, recommended = self.get_ollama_context_range()
        range_text = (
            f"{range_min:,}~{range_max:,} (권장 {recommended:,})"
            if range_max
            else f"{range_min:,}+ (권장 {recommended:,}, 모델 최대 미감지)"
        )
        raw = str(self.ollama_context_input.text() or "").strip()
        if not raw:
            self.set_ollama_context_hint(
                f"⚠ Context 값이 비어 있습니다. 권장 범위: {range_text}",
                "#ef4444",
            )
            if show_popup:
                QMessageBox.warning(
                    self,
                    "Context 설정 필요",
                    f"Ollama Context를 입력하세요.\n권장 범위: {range_text}",
                )
            return False
        try:
            value = int(raw)
        except Exception:
            self.set_ollama_context_hint(
                "⚠ Context 값은 숫자여야 합니다.",
                "#ef4444",
            )
            if show_popup:
                QMessageBox.warning(self, "Context 형식 오류", "Ollama Context는 숫자로 입력해야 합니다.")
            return False

        if value < range_min:
            reason = (
                "OpenClaw 최소 요구치 미만이라 요청이 시작되기 전에 실패합니다 "
                f"(FailoverError: Model context window too small, minimum {OPENCLAW_MIN_CONTEXT_WINDOW})."
            )
            self.set_ollama_context_hint(
                f"⚠ 현재 {value:,} < 최소 {range_min:,}. 권장 범위: {range_text}. {reason}",
                "#ef4444",
            )
            if show_popup:
                QMessageBox.warning(self, "Context 부족", reason)
            return False

        if range_max and value > range_max:
            self.set_ollama_context_hint(
                f"⚠ 현재 {value:,} > 모델 감지 최대 {range_max:,}. 권장 범위: {range_text} "
                "(실행 시 내부 조정되거나 실패할 수 있습니다.)",
                "#f59e0b",
            )
            return True

        source_text = self.detected_context_source or "manual"
        self.set_ollama_context_hint(
            f"✅ 현재 {value:,}. 권장 범위: {range_text} (감지 출처: {source_text})",
            "#22c55e",
        )
        return True

    def detect_ollama_context(self, show_popup: bool = False):
        if self.current_source() != self.PROVIDER_OLLAMA:
            return
        if self.ollama_context_thread and self.ollama_context_thread.isRunning():
            return
        base_url = str(self.ollama_base_url_input.text() or "").strip() or "http://127.0.0.1:11434/v1"
        model_name = str(self.ollama_model_input.text() or "").strip()
        if not model_name:
            self.set_ollama_context_hint("⚠ 모델 이름이 비어 있어 자동 감지를 건너뜁니다.", "#f59e0b")
            return
        self._context_popup_pending = bool(show_popup)
        self.set_ollama_context_hint("Ollama 컨텍스트 자동 감지 중...", "#f59e0b")
        if hasattr(self, "btn_ollama_detect_context"):
            self.btn_ollama_detect_context.setEnabled(False)
        self.ollama_context_thread = OllamaContextDetectThread(base_url, model_name)
        self.ollama_context_thread.completed.connect(self.on_ollama_context_detect_completed)
        self.ollama_context_thread.start()

    def on_ollama_context_detect_completed(self, ok: bool, context_window: int, source: str):
        if ok:
            detected = clamp_int(
                context_window,
                DEFAULT_OLLAMA_CONTEXT_WINDOW,
                OPENCLAW_MIN_CONTEXT_WINDOW,
                MAX_OLLAMA_CONTEXT_WINDOW,
            )
            self.detected_context_window = detected
            self.detected_context_source = str(source or "")
            range_min, range_max, recommended = self.get_ollama_context_range()
            raw_current = str(self.ollama_context_input.text() or "").strip()
            try:
                current_value = int(raw_current)
            except Exception:
                current_value = recommended
            if current_value < range_min:
                self.ollama_context_input.setText(str(recommended))
            self.validate_ollama_context(show_popup=False)
            if self._context_popup_pending:
                QMessageBox.information(
                    self,
                    "Context 자동 감지",
                    f"모델 컨텍스트를 감지했습니다: {detected:,}\n"
                    f"권장 범위: {range_min:,}~{range_max:,} (권장 {recommended:,})\n"
                    f"출처: {source}",
                )
        else:
            self.detected_context_window = None
            self.detected_context_source = ""
            self.set_ollama_context_hint(
                "⚠ 자동 감지 실패. 수동 입력을 사용합니다. "
                f"(사유: {source or 'unknown'})",
                "#f59e0b",
            )
            self.validate_ollama_context(show_popup=False)
            if self._context_popup_pending:
                QMessageBox.warning(
                    self,
                    "Context 자동 감지 실패",
                    source or "모델 메타데이터에서 컨텍스트 길이를 찾지 못했습니다.",
                )

        self._context_popup_pending = False
        if hasattr(self, "btn_ollama_detect_context"):
            self.btn_ollama_detect_context.setEnabled(True)
        self.ollama_context_thread = None

    def on_source_changed(self):
        if self._profile_loading:
            return
        self.sync_source_visibility()
        if self.current_source() == self.PROVIDER_OLLAMA:
            self.detect_ollama_tools_support(show_popup=False)
        elif self.current_source() == self.PROVIDER_API:
            self.sync_api_model_preview()
        self.check_auth()

    def sync_source_visibility(self):
        source = self.current_source()
        use_vertex = source == self.PROVIDER_VERTEX
        use_ollama = source == self.PROVIDER_OLLAMA
        use_api = source == self.PROVIDER_API
        self.model_combo.setEnabled(use_vertex)
        self.vertex_sa_row.setVisible(use_vertex)
        self.vertex_project_row.setVisible(use_vertex)
        self.vertex_location_row.setVisible(use_vertex)
        self.advanced_row.setVisible(use_ollama or use_api)
        self.ollama_base_row.setVisible(use_ollama)
        self.ollama_model_row.setVisible(use_ollama)
        self.ollama_models_row.setVisible(use_ollama)
        self.ollama_context_row.setVisible(use_ollama)
        self.ollama_injection_row.setVisible(use_ollama)
        self.ollama_history_row.setVisible(use_ollama)
        self.api_preset_row.setVisible(use_api)
        self.api_provider_row.setVisible(use_api)
        self.api_base_row.setVisible(use_api)
        self.api_model_row.setVisible(use_api)
        self.api_models_row.setVisible(use_api)
        self.api_context_row.setVisible(use_api)
        show_advanced = bool(self.show_advanced_checkbox.isChecked())
        self.ollama_api_row.setVisible(use_ollama and show_advanced)
        self.api_key_row.setVisible(use_api)
        self.api_format_row.setVisible(use_api and show_advanced)

        if use_vertex:
            self.model_label.setText("BRAIN 모델")
            self.note_label.setText(
                "Vertex 모드: Service Account JSON / Project / Location 인증을 사용합니다."
            )
            self.set_ollama_context_hint("")
            self.update_tools_support_hint()
            return

        self.model_label.setText("실행 모델(OpenClaw ID)")
        if use_ollama:
            effective = self.normalize_ollama_model(self.ollama_model_input.text())
            self.model_combo.setCurrentText(effective)
            self.update_tools_support_hint()
            if show_advanced:
                self.note_label.setText(
                    "Ollama 모드: 목록은 참고용이며, 실제 실행값은 Ollama Model/실행 모델(OpenClaw ID)입니다. "
                    "주입 강도/히스토리 턴 수로 대화 품질과 지연을 조정할 수 있습니다."
                )
            else:
                self.note_label.setText(
                    "Ollama 모드: 목록은 참고용이며, 실제 실행값은 Ollama Model/실행 모델(OpenClaw ID)입니다. "
                    "주입 강도/히스토리 턴 수로 품질을 조절하고, API Key는 고급 옵션에서 필요할 때만 설정하세요."
                )
            self.validate_ollama_context(show_popup=False)
            return

        self.sync_api_model_preview()
        self.update_tools_support_hint()
        if show_advanced:
            self.note_label.setText(
                "API 모드: 프리셋 또는 Provider ID + Base URL + Model 조합으로 OpenClaw custom provider를 등록합니다."
            )
            return
        self.note_label.setText(
            "API 모드: 프리셋(OpenAI/OpenRouter/Groq 등) 선택 후 필요시 수동 수정하세요. "
            "일반적으로 Provider ID, Base URL, Model만 지정하면 됩니다. "
            "API Key/포맷은 고급 옵션에서 설정하세요."
        )

    def sync_ollama_model_preview(self):
        if self.current_source() != self.PROVIDER_OLLAMA:
            return
        self.model_combo.setCurrentText(self.normalize_ollama_model(self.ollama_model_input.text()))

    def on_ollama_history_turns_changed(self, value: int):
        turns = clamp_int(value, 6, 1, 20)
        if hasattr(self, "ollama_history_value_label"):
            self.ollama_history_value_label.setText(f"{turns}턴")

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        profile_row = QHBoxLayout()
        profile_label = QLabel("프로필")
        self.profile_combo = QComboBox()
        self.btn_profile_add = QPushButton("+")
        self.btn_profile_add.setFixedWidth(28)
        self.btn_profile_remove = QPushButton("-")
        self.btn_profile_remove.setFixedWidth(28)
        self.btn_profile_add.clicked.connect(self.create_profile)
        self.btn_profile_remove.clicked.connect(self.delete_current_profile)
        self.profile_combo.currentIndexChanged.connect(self.on_profile_changed)
        profile_row.addWidget(profile_label)
        profile_row.addWidget(self.profile_combo, 1)
        profile_row.addWidget(self.btn_profile_add)
        profile_row.addWidget(self.btn_profile_remove)
        layout.addLayout(profile_row)

        lang_row = QHBoxLayout()
        lang_label = QLabel("UI 언어")
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("한국어", "ko")
        self.lang_combo.addItem("English", "en")
        ui_settings = self.settings.get("ui", {}) if isinstance(self.settings.get("ui"), dict) else {}
        current_lang = str(ui_settings.get("language") or "ko").strip().lower()
        lang_index = self.lang_combo.findData(current_lang)
        self.lang_combo.setCurrentIndex(lang_index if lang_index >= 0 else 0)
        lang_row.addWidget(lang_label)
        lang_row.addWidget(self.lang_combo)
        layout.addLayout(lang_row)

        agent_row = QHBoxLayout()
        agent_row.setContentsMargins(0, 0, 0, 0)
        agent_row.addWidget(QLabel("Agent ID"))
        self.agent_id_input = QLineEdit(str(self.profile_settings.get("agentId") or self.active_profile_id))
        agent_row.addWidget(self.agent_id_input)
        layout.addLayout(agent_row)

        workspace_row = QHBoxLayout()
        workspace_row.setContentsMargins(0, 0, 0, 0)
        workspace_row.addWidget(QLabel("Workspace"))
        self.workspace_input = QLineEdit(str(self.profile_settings.get("workspace") or ""))
        workspace_row.addWidget(self.workspace_input)
        self.workspace_browse_btn = QPushButton("찾기")
        self.workspace_browse_btn.clicked.connect(self.browse_workspace)
        workspace_row.addWidget(self.workspace_browse_btn)
        layout.addLayout(workspace_row)

        session_row = QHBoxLayout()
        session_row.setContentsMargins(0, 0, 0, 0)
        session_row.addWidget(QLabel("Session Key"))
        self.session_key_input = QLineEdit(str(self.profile_settings.get("sessionKey") or ""))
        self.session_key_input.setPlaceholderText("비워두면 런타임 세션키 사용")
        session_row.addWidget(self.session_key_input)
        layout.addLayout(session_row)

        source_row = QHBoxLayout()
        source_label = QLabel("모델 소스")
        self.source_combo = QComboBox()
        self.source_combo.addItem("Vertex API", self.PROVIDER_VERTEX)
        self.source_combo.addItem("로컬 GPU (Ollama)", self.PROVIDER_OLLAMA)
        self.source_combo.addItem("범용 API (OpenAI 호환)", self.PROVIDER_API)
        current_source = str(self.profile_settings.get("modelSource") or self.PROVIDER_VERTEX).strip().lower()
        source_index = self.source_combo.findData(current_source)
        self.source_combo.setCurrentIndex(source_index if source_index >= 0 else 0)
        source_row.addWidget(source_label)
        source_row.addWidget(self.source_combo)
        layout.addLayout(source_row)

        tool_mode_row = QHBoxLayout()
        tool_mode_label = QLabel("대화 모드")
        self.tool_mode_combo = QComboBox()
        self.tool_mode_combo.addItem("자동(모델 기준)", "auto")
        self.tool_mode_combo.addItem("대화 전용 (도구 OFF)", "chat")
        self.tool_mode_combo.addItem("에이전트 (도구 ON)", "agent")
        current_tool_mode = str(self.profile_settings.get("toolMode") or "auto").strip().lower()
        tool_mode_index = self.tool_mode_combo.findData(
            current_tool_mode if current_tool_mode in {"auto", "chat", "agent"} else "auto"
        )
        self.tool_mode_combo.setCurrentIndex(tool_mode_index if tool_mode_index >= 0 else 0)
        tool_mode_row.addWidget(tool_mode_label)
        tool_mode_row.addWidget(self.tool_mode_combo)
        layout.addLayout(tool_mode_row)
        self.tool_support_hint = QLabel("")
        self.tool_support_hint.setStyleSheet("color: #9ca3af; font-size: 11px;")
        layout.addWidget(self.tool_support_hint)

        model_row = QHBoxLayout()
        self.model_label = QLabel("BRAIN 모델")
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems([
            "google-vertex/gemini-2.0-flash",
            "google-vertex/gemini-1.5-pro",
            "openai/gpt-4.1-mini",
            "openrouter/openai/gpt-4.1-mini",
            "ollama/llama3.3",
        ])
        current_model = self.profile_settings.get("model") or "google-vertex/gemini-2.0-flash"
        if self.model_combo.findText(current_model) == -1:
            self.model_combo.addItem(current_model)
        self.model_combo.setCurrentText(current_model)
        model_row.addWidget(self.model_label)
        model_row.addWidget(self.model_combo)
        layout.addLayout(model_row)

        self.vertex_sa_row = QWidget()
        sa_row = QHBoxLayout(self.vertex_sa_row)
        sa_row.setContentsMargins(0, 0, 0, 0)
        sa_label = QLabel("Vertex SA JSON")
        vertex_settings = (
            self.profile_settings.get("vertex", {})
            if isinstance(self.profile_settings.get("vertex"), dict)
            else {}
        )
        self.sa_input = QLineEdit(vertex_settings.get("serviceAccountPath", ""))
        btn_browse = QPushButton("찾기")
        btn_browse.clicked.connect(self.browse_sa)
        sa_row.addWidget(sa_label)
        sa_row.addWidget(self.sa_input)
        sa_row.addWidget(btn_browse)
        layout.addWidget(self.vertex_sa_row)

        self.vertex_project_row = QWidget()
        proj_row = QHBoxLayout(self.vertex_project_row)
        proj_row.setContentsMargins(0, 0, 0, 0)
        proj_label = QLabel("Vertex Project")
        self.project_input = QLineEdit(vertex_settings.get("project", ""))
        proj_row.addWidget(proj_label)
        proj_row.addWidget(self.project_input)
        layout.addWidget(self.vertex_project_row)

        self.vertex_location_row = QWidget()
        loc_row = QHBoxLayout(self.vertex_location_row)
        loc_row.setContentsMargins(0, 0, 0, 0)
        loc_label = QLabel("Vertex Location")
        self.location_input = QLineEdit(vertex_settings.get("location", ""))
        loc_row.addWidget(loc_label)
        loc_row.addWidget(self.location_input)
        layout.addWidget(self.vertex_location_row)

        ollama_settings = (
            self.profile_settings.get("ollama", {})
            if isinstance(self.profile_settings.get("ollama"), dict)
            else {}
        )
        default_ollama_name = str(ollama_settings.get("modelName") or "").strip()
        if not default_ollama_name and str(current_model).startswith("ollama/"):
            default_ollama_name = str(current_model).split("/", 1)[1]
        if not default_ollama_name:
            default_ollama_name = "llama3.3"

        self.ollama_base_row = QWidget()
        ollama_base_row_layout = QHBoxLayout(self.ollama_base_row)
        ollama_base_row_layout.setContentsMargins(0, 0, 0, 0)
        ollama_base_label = QLabel("Ollama Base URL")
        self.ollama_base_url_input = QLineEdit(
            str(ollama_settings.get("baseUrl") or "http://127.0.0.1:11434/v1")
        )
        ollama_base_row_layout.addWidget(ollama_base_label)
        ollama_base_row_layout.addWidget(self.ollama_base_url_input)
        layout.addWidget(self.ollama_base_row)

        self.ollama_model_row = QWidget()
        ollama_model_row_layout = QHBoxLayout(self.ollama_model_row)
        ollama_model_row_layout.setContentsMargins(0, 0, 0, 0)
        ollama_model_label = QLabel("Ollama Model")
        self.ollama_model_input = QLineEdit(default_ollama_name)
        self.ollama_model_input.setPlaceholderText("예: llama3.3 또는 qwen2.5-coder:32b")
        ollama_model_row_layout.addWidget(ollama_model_label)
        ollama_model_row_layout.addWidget(self.ollama_model_input)
        self.btn_ollama_refresh = QPushButton("목록 불러오기")
        self.btn_ollama_refresh.clicked.connect(lambda: self.load_ollama_models(show_popup=True))
        ollama_model_row_layout.addWidget(self.btn_ollama_refresh)
        layout.addWidget(self.ollama_model_row)

        self.ollama_models_row = QWidget()
        ollama_models_row_layout = QHBoxLayout(self.ollama_models_row)
        ollama_models_row_layout.setContentsMargins(0, 0, 0, 0)
        ollama_models_label = QLabel("Ollama 목록")
        self.ollama_models_combo = QComboBox()
        self.ollama_models_combo.setEnabled(False)
        self.ollama_models_combo.addItem("(모델 없음)")
        self.ollama_models_combo.currentTextChanged.connect(self.on_ollama_model_selected)
        ollama_models_row_layout.addWidget(ollama_models_label)
        ollama_models_row_layout.addWidget(self.ollama_models_combo)
        layout.addWidget(self.ollama_models_row)

        self.ollama_context_row = QWidget()
        ollama_context_row_layout = QHBoxLayout(self.ollama_context_row)
        ollama_context_row_layout.setContentsMargins(0, 0, 0, 0)
        ollama_context_label = QLabel("Ollama Context(토큰)")
        self.ollama_context_input = QLineEdit(
            str(
                clamp_int(
                    ollama_settings.get("contextWindow"),
                    DEFAULT_OLLAMA_CONTEXT_WINDOW,
                    OPENCLAW_MIN_CONTEXT_WINDOW,
                    MAX_OLLAMA_CONTEXT_WINDOW,
                )
            )
        )
        self.ollama_context_input.setPlaceholderText(
            f"최소 {OPENCLAW_MIN_CONTEXT_WINDOW} (권장 {DEFAULT_OLLAMA_CONTEXT_WINDOW})"
        )
        ollama_context_row_layout.addWidget(ollama_context_label)
        ollama_context_row_layout.addWidget(self.ollama_context_input)
        self.btn_ollama_detect_context = QPushButton("자동 감지")
        self.btn_ollama_detect_context.clicked.connect(lambda: self.detect_ollama_context(show_popup=True))
        ollama_context_row_layout.addWidget(self.btn_ollama_detect_context)
        layout.addWidget(self.ollama_context_row)
        self.ollama_context_hint = QLabel("")
        self.ollama_context_hint.setStyleSheet("color: #9ca3af; font-size: 11px;")
        layout.addWidget(self.ollama_context_hint)

        self.ollama_injection_row = QWidget()
        ollama_injection_row_layout = QHBoxLayout(self.ollama_injection_row)
        ollama_injection_row_layout.setContentsMargins(0, 0, 0, 0)
        ollama_injection_label = QLabel("주입 강도")
        self.ollama_injection_combo = QComboBox()
        self.ollama_injection_combo.addItem("약", "weak")
        self.ollama_injection_combo.addItem("중", "medium")
        self.ollama_injection_combo.addItem("강", "strong")
        current_injection_level = str(ollama_settings.get("injectionLevel") or "medium").strip().lower()
        injection_index = self.ollama_injection_combo.findData(
            current_injection_level if current_injection_level in {"weak", "medium", "strong"} else "medium"
        )
        self.ollama_injection_combo.setCurrentIndex(injection_index if injection_index >= 0 else 1)
        ollama_injection_row_layout.addWidget(ollama_injection_label)
        ollama_injection_row_layout.addWidget(self.ollama_injection_combo)
        layout.addWidget(self.ollama_injection_row)

        self.ollama_history_row = QWidget()
        ollama_history_row_layout = QHBoxLayout(self.ollama_history_row)
        ollama_history_row_layout.setContentsMargins(0, 0, 0, 0)
        ollama_history_label = QLabel("히스토리 턴 수")
        self.ollama_history_slider = QSlider(Qt.Horizontal)
        self.ollama_history_slider.setRange(1, 20)
        self.ollama_history_slider.setSingleStep(1)
        self.ollama_history_slider.setPageStep(1)
        self.ollama_history_slider.setTickPosition(QSlider.NoTicks)
        current_history_turns = clamp_int(ollama_settings.get("historyTurns"), 6, 1, 20)
        self.ollama_history_slider.setValue(current_history_turns)
        self.ollama_history_value_label = QLabel("")
        self.ollama_history_value_label.setMinimumWidth(42)
        self.on_ollama_history_turns_changed(current_history_turns)
        ollama_history_row_layout.addWidget(ollama_history_label)
        ollama_history_row_layout.addWidget(self.ollama_history_slider)
        ollama_history_row_layout.addWidget(self.ollama_history_value_label)
        layout.addWidget(self.ollama_history_row)

        self.advanced_row = QWidget()
        advanced_row_layout = QHBoxLayout(self.advanced_row)
        advanced_row_layout.setContentsMargins(0, 0, 0, 0)
        advanced_label = QLabel("고급 옵션")
        self.show_advanced_checkbox = QCheckBox("표시")
        self.show_advanced_checkbox.setChecked(False)
        self.show_advanced_checkbox.stateChanged.connect(self.sync_source_visibility)
        advanced_row_layout.addWidget(advanced_label)
        advanced_row_layout.addWidget(self.show_advanced_checkbox)
        advanced_row_layout.addStretch()
        layout.addWidget(self.advanced_row)

        self.ollama_api_row = QWidget()
        ollama_api_row_layout = QHBoxLayout(self.ollama_api_row)
        ollama_api_row_layout.setContentsMargins(0, 0, 0, 0)
        ollama_api_label = QLabel("Ollama API Key (옵션)")
        self.ollama_api_key_input = QLineEdit(str(ollama_settings.get("apiKey") or "ollama-local"))
        self.ollama_api_key_input.setPlaceholderText("로컬 기본은 비워도 됩니다 (자동값: ollama-local)")
        ollama_api_row_layout.addWidget(ollama_api_label)
        ollama_api_row_layout.addWidget(self.ollama_api_key_input)
        layout.addWidget(self.ollama_api_row)

        api_settings = (
            self.profile_settings.get("apiProvider", {})
            if isinstance(self.profile_settings.get("apiProvider"), dict)
            else {}
        )
        default_api_provider_id = normalize_provider_id(str(api_settings.get("providerId") or ""), "openai")
        default_api_model_name = str(api_settings.get("modelName") or "").strip() or "gpt-4.1-mini"
        default_api_base_url = str(api_settings.get("baseUrl") or "").strip() or "https://api.openai.com/v1"
        default_api_format = str(api_settings.get("apiFormat") or "openai-completions").strip().lower()
        if default_api_format not in {"openai-completions", "openai-responses", "anthropic-messages"}:
            default_api_format = "openai-completions"
        initial_preset_key = self.infer_api_preset_key(default_api_provider_id, default_api_base_url)

        self.api_preset_row = QWidget()
        api_preset_row_layout = QHBoxLayout(self.api_preset_row)
        api_preset_row_layout.setContentsMargins(0, 0, 0, 0)
        api_preset_row_layout.addWidget(QLabel("API 프리셋"))
        self.api_preset_combo = QComboBox()
        for preset_key, preset in self.API_PROVIDER_PRESETS.items():
            self.api_preset_combo.addItem(str(preset.get("label") or preset_key), preset_key)
        self.api_preset_combo.addItem("Custom (직접 입력)", "custom")
        preset_index = self.api_preset_combo.findData(initial_preset_key)
        self.api_preset_combo.setCurrentIndex(preset_index if preset_index >= 0 else 0)
        api_preset_row_layout.addWidget(self.api_preset_combo)
        self.btn_apply_api_preset = QPushButton("적용")
        self.btn_apply_api_preset.clicked.connect(self.on_api_preset_changed)
        api_preset_row_layout.addWidget(self.btn_apply_api_preset)
        layout.addWidget(self.api_preset_row)

        self.api_provider_row = QWidget()
        api_provider_row_layout = QHBoxLayout(self.api_provider_row)
        api_provider_row_layout.setContentsMargins(0, 0, 0, 0)
        api_provider_row_layout.addWidget(QLabel("API Provider ID"))
        self.api_provider_id_input = QLineEdit(default_api_provider_id)
        self.api_provider_id_input.setPlaceholderText("예: openai, openrouter, lmstudio")
        api_provider_row_layout.addWidget(self.api_provider_id_input)
        layout.addWidget(self.api_provider_row)

        self.api_base_row = QWidget()
        api_base_row_layout = QHBoxLayout(self.api_base_row)
        api_base_row_layout.setContentsMargins(0, 0, 0, 0)
        api_base_row_layout.addWidget(QLabel("API Base URL"))
        self.api_base_url_input = QLineEdit(default_api_base_url)
        self.api_base_url_input.setPlaceholderText("예: https://api.openai.com/v1")
        api_base_row_layout.addWidget(self.api_base_url_input)
        layout.addWidget(self.api_base_row)

        self.api_model_row = QWidget()
        api_model_row_layout = QHBoxLayout(self.api_model_row)
        api_model_row_layout.setContentsMargins(0, 0, 0, 0)
        api_model_row_layout.addWidget(QLabel("API Model"))
        self.api_model_input = QLineEdit(default_api_model_name)
        self.api_model_input.setPlaceholderText("예: gpt-4.1-mini, anthropic/claude-sonnet-4-5")
        api_model_row_layout.addWidget(self.api_model_input)
        self.btn_api_refresh = QPushButton("목록 불러오기")
        self.btn_api_refresh.clicked.connect(lambda: self.load_api_models(show_popup=True))
        api_model_row_layout.addWidget(self.btn_api_refresh)
        layout.addWidget(self.api_model_row)

        self.api_models_row = QWidget()
        api_models_row_layout = QHBoxLayout(self.api_models_row)
        api_models_row_layout.setContentsMargins(0, 0, 0, 0)
        api_models_row_layout.addWidget(QLabel("API 목록"))
        self.api_models_combo = QComboBox()
        self.api_models_combo.setEnabled(False)
        self.api_models_combo.addItem("(모델 없음)")
        self.api_models_combo.currentTextChanged.connect(self.on_api_model_selected)
        api_models_row_layout.addWidget(self.api_models_combo)
        layout.addWidget(self.api_models_row)

        self.api_context_row = QWidget()
        api_context_row_layout = QHBoxLayout(self.api_context_row)
        api_context_row_layout.setContentsMargins(0, 0, 0, 0)
        api_context_row_layout.addWidget(QLabel("API Context(토큰)"))
        self.api_context_input = QLineEdit(
            str(
                clamp_int(
                    api_settings.get("contextWindow"),
                    DEFAULT_API_CONTEXT_WINDOW,
                    OPENCLAW_MIN_CONTEXT_WINDOW,
                    MAX_API_CONTEXT_WINDOW,
                )
            )
        )
        self.api_context_input.setPlaceholderText(
            f"최소 {OPENCLAW_MIN_CONTEXT_WINDOW} (권장 {DEFAULT_API_CONTEXT_WINDOW})"
        )
        api_context_row_layout.addWidget(self.api_context_input)
        layout.addWidget(self.api_context_row)

        self.api_format_row = QWidget()
        api_format_row_layout = QHBoxLayout(self.api_format_row)
        api_format_row_layout.setContentsMargins(0, 0, 0, 0)
        api_format_row_layout.addWidget(QLabel("API Format"))
        self.api_format_combo = QComboBox()
        self.api_format_combo.addItem("OpenAI Chat Completions", "openai-completions")
        self.api_format_combo.addItem("OpenAI Responses", "openai-responses")
        self.api_format_combo.addItem("Anthropic Messages", "anthropic-messages")
        api_format_index = self.api_format_combo.findData(default_api_format)
        self.api_format_combo.setCurrentIndex(api_format_index if api_format_index >= 0 else 0)
        api_format_row_layout.addWidget(self.api_format_combo)
        layout.addWidget(self.api_format_row)

        self.api_key_row = QWidget()
        api_key_row_layout = QHBoxLayout(self.api_key_row)
        api_key_row_layout.setContentsMargins(0, 0, 0, 0)
        api_key_row_layout.addWidget(QLabel("API Key"))
        self.api_key_input = QLineEdit(str(api_settings.get("apiKey") or ""))
        self.api_key_input.setPlaceholderText("sk-... 또는 provider API key")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        api_key_row_layout.addWidget(self.api_key_input)
        layout.addWidget(self.api_key_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.auth_status = QLabel("인증 상태: 미확인 ●")
        self.auth_status.setStyleSheet("color: #9ca3af; font-size: 11px;")
        btn_check = QPushButton("인증 확인")
        btn_check.clicked.connect(self.check_auth)
        self.btn_check_auth = btn_check
        btn_apply = QPushButton("저장/적용")
        btn_cancel = QPushButton("취소")
        btn_apply.clicked.connect(self.apply_settings)
        btn_cancel.clicked.connect(self.reject)
        self.btn_apply = btn_apply
        self.btn_cancel = btn_cancel
        btn_row.addWidget(self.auth_status)
        btn_row.addWidget(btn_check)
        btn_row.addWidget(btn_apply)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        self.note_label = QLabel("")
        self.note_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        layout.addWidget(self.note_label)
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        self.tool_mode_combo.currentIndexChanged.connect(self.on_tool_mode_changed)
        self.ollama_base_url_input.editingFinished.connect(self.check_auth)
        self.ollama_base_url_input.textChanged.connect(self.sync_source_visibility)
        self.ollama_model_input.textChanged.connect(self.sync_ollama_model_preview)
        self.ollama_model_input.editingFinished.connect(lambda: self.detect_ollama_context(show_popup=False))
        self.ollama_model_input.editingFinished.connect(lambda: self.detect_ollama_tools_support(show_popup=False))
        self.ollama_context_input.textChanged.connect(lambda: self.validate_ollama_context(show_popup=False))
        self.ollama_history_slider.valueChanged.connect(self.on_ollama_history_turns_changed)
        self.api_provider_id_input.editingFinished.connect(self.sync_api_model_preview)
        self.api_provider_id_input.editingFinished.connect(self.sync_api_preset_from_inputs)
        self.api_base_url_input.editingFinished.connect(self.sync_api_preset_from_inputs)
        self.api_base_url_input.editingFinished.connect(self.check_auth)
        self.api_model_input.textChanged.connect(self.sync_api_model_preview)
        self.api_model_input.editingFinished.connect(self.check_auth)
        self.api_key_input.editingFinished.connect(self.check_auth)
        self.api_preset_combo.currentIndexChanged.connect(self.on_api_preset_changed)
        self.refresh_profile_selector(self.active_profile_id)
        self.load_profile_into_form(self.active_profile_id)

    def browse_sa(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Vertex Service Account JSON 선택", "", "JSON Files (*.json)"
        )
        if path:
            self.sa_input.setText(path)

    def browse_workspace(self):
        path = QFileDialog.getExistingDirectory(self, "워크스페이스 폴더 선택")
        if path:
            self.workspace_input.setText(path)

    def run_openclaw(self, openclaw_mjs: Path, args: list[str], timeout: int = 20) -> tuple[int, str, str]:
        return run_process_capture(
            ["node", str(openclaw_mjs), *args],
            cwd=str(openclaw_mjs.parent),
            timeout=timeout,
        )

    def read_agents_list(self, openclaw_mjs: Path) -> list[dict]:
        code, stdout_text, _ = self.run_openclaw(openclaw_mjs, ["config", "get", "--json", "agents.list"])
        if code != 0:
            return []
        try:
            parsed = json.loads(stdout_text or "[]")
        except Exception:
            return []
        if isinstance(parsed, list):
            return [entry for entry in parsed if isinstance(entry, dict)]
        return []

    def ensure_profile_agent_config(self, openclaw_mjs: Path, profile: dict) -> list[str]:
        errors: list[str] = []
        profile_id = normalize_profile_id(str(self.profile_combo.currentData() or self.active_profile_id))
        agent_id = normalize_agent_id(str(profile.get("agentId") or profile_id))
        workspace = str(profile.get("workspace") or "").strip()
        model = str(profile.get("model") or "").strip()
        if not model:
            model = "google-vertex/gemini-2.0-flash"
        if not workspace:
            workspace = str(default_profile(profile_id).get("workspace") or "")

        agents = self.read_agents_list(openclaw_mjs)
        index = -1
        for idx, entry in enumerate(agents):
            if normalize_agent_id(str(entry.get("id") or "")) == agent_id:
                index = idx
                break
        if index < 0:
            index = len(agents)

        updates = [
            (f"agents.list[{index}].id", agent_id),
            (f"agents.list[{index}].workspace", workspace),
            (f"agents.list[{index}].model.primary", model),
        ]
        for path, value in updates:
            code, stdout_text, stderr_text = self.run_openclaw(
                openclaw_mjs,
                ["config", "set", path, value],
            )
            if code != 0:
                errors.append(f"{path}: {stderr_text.strip() or stdout_text.strip()}")
        return errors

    def apply_settings(self):
        if self.apply_thread and self.apply_thread.isRunning():
            return
        source = self.current_source()
        if source == self.PROVIDER_OLLAMA:
            if not self.validate_ollama_context(show_popup=True):
                return
            if not self.enforce_tool_mode_compatibility(show_popup=True):
                return
        elif source == self.PROVIDER_API:
            provider_id = self.get_api_provider_id()
            model_name = str(self.api_model_input.text() or "").strip()
            base_url = str(self.api_base_url_input.text() or "").strip()
            api_key = str(self.api_key_input.text() or "").strip()
            if not provider_id or not model_name or not base_url:
                QMessageBox.warning(
                    self,
                    "입력 확인",
                    "API 모드에서는 Provider ID, Base URL, API Model을 모두 입력해야 합니다.",
                )
                return
            if not api_key:
                QMessageBox.warning(
                    self,
                    "입력 확인",
                    "API 모드에서는 API Key를 입력해야 합니다.",
                )
                return
        selected_profile_id = normalize_profile_id(
            str(self.profile_combo.currentData() or self.active_profile_id)
        )
        profile_settings = self.build_pending_profile(selected_profile_id)
        prepared_workspace = ensure_profile_workspace_scaffold(
            selected_profile_id,
            str(profile_settings.get("workspace") or ""),
        )
        profile_settings["workspace"] = str(prepared_workspace)
        save_settings(self.build_storage_settings(selected_profile_id))
        self.settings = load_settings()
        self.active_profile_id = selected_profile_id
        self.profile_settings = get_profile(self.settings, self.active_profile_id)
        self.set_apply_controls_enabled(False)
        self.set_auth_status(None, "적용 중")
        self.note_label.setText("OpenClaw 적용 중... 잠시만 기다려주세요.")
        self.apply_thread = ApplySettingsThread(selected_profile_id, profile_settings)
        self.apply_thread.completed.connect(self.on_apply_completed)
        self.apply_thread.start()

    def set_apply_controls_enabled(self, enabled: bool):
        enabled_state = bool(enabled)
        for widget_name in (
            "btn_apply",
            "btn_cancel",
            "btn_check_auth",
            "btn_ollama_refresh",
            "btn_ollama_detect_context",
            "btn_api_refresh",
            "btn_apply_api_preset",
            "source_combo",
            "tool_mode_combo",
            "ollama_injection_combo",
            "ollama_history_slider",
            "api_preset_combo",
            "api_format_combo",
            "profile_combo",
            "btn_profile_add",
            "btn_profile_remove",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(enabled_state)

    def on_apply_completed(self, errors_obj: object):
        errors = [str(item) for item in errors_obj] if isinstance(errors_obj, list) else []
        self.set_apply_controls_enabled(True)
        self.settings = load_settings()
        self.profile_settings = get_profile(self.settings, self.active_profile_id)
        self.apply_thread = None

        if errors:
            QMessageBox.warning(
                self,
                "설정 저장(일부 적용 실패)",
                "설정은 저장되었지만 OpenClaw 반영 중 일부 실패했습니다.\n\n"
                + "\n".join(errors[:4]),
            )
            self.accept()
            return

        QMessageBox.information(
            self,
            "설정 저장됨",
            "설정이 저장되었습니다.\n프로필별 워크스페이스/페르소나 파일 분리도 적용되었습니다.",
        )
        self.accept()

    def on_auth_check_completed(self, ok: bool, label: str, models_obj: object, detail: str):
        models = models_obj if isinstance(models_obj, list) else []
        source = self.current_source()
        if source == self.PROVIDER_OLLAMA:
            self.set_ollama_models([str(item) for item in models if isinstance(item, str)])
            if ok:
                self.detect_ollama_context(show_popup=False)
                self.detect_ollama_tools_support(show_popup=False)
            else:
                self.validate_ollama_context(show_popup=False)
                self.detected_tools_supported = None
                self.detected_tools_detail = detail or "Ollama 연결 실패"
                self.update_tools_support_hint()
        elif source == self.PROVIDER_API:
            self.set_api_models([str(item) for item in models if isinstance(item, str)])
            if ok:
                self.sync_api_model_preview()
        self.set_auth_status(ok, label)

        if self._auth_popup_pending:
            if ok:
                if source == self.PROVIDER_OLLAMA:
                    QMessageBox.information(self, "Ollama 연결", f"모델 {len(models)}개를 불러왔습니다.")
                elif source == self.PROVIDER_API:
                    QMessageBox.information(self, "API 연결", f"모델 {len(models)}개를 불러왔습니다.")
                else:
                    QMessageBox.information(self, "인증 확인", "Vertex 인증이 정상입니다.")
            else:
                message = detail.strip() if detail else "인증 확인에 실패했습니다."
                if source == self.PROVIDER_OLLAMA:
                    title = "Ollama 연결 실패"
                elif source == self.PROVIDER_API:
                    title = "API 연결 실패"
                else:
                    title = "인증 실패"
                QMessageBox.warning(self, title, message)

        self._auth_popup_pending = False
        if hasattr(self, "btn_check_auth"):
            self.btn_check_auth.setEnabled(True)
        if hasattr(self, "btn_ollama_refresh"):
            self.btn_ollama_refresh.setEnabled(True)
        if hasattr(self, "btn_api_refresh"):
            self.btn_api_refresh.setEnabled(True)
        self.auth_check_thread = None

    def check_auth(self, checked: bool = False, show_popup: bool = False):
        _ = checked
        if self._profile_loading:
            return
        if self.auth_check_thread and self.auth_check_thread.isRunning():
            return

        pending = self.build_pending_profile(
            str(self.profile_combo.currentData() or self.active_profile_id)
        )
        if self.current_source() == self.PROVIDER_OLLAMA:
            self.validate_ollama_context(show_popup=False)
        self._auth_popup_pending = bool(show_popup)
        self.set_auth_status(None, "확인 중")
        if hasattr(self, "btn_check_auth"):
            self.btn_check_auth.setEnabled(False)
        if hasattr(self, "btn_ollama_refresh"):
            self.btn_ollama_refresh.setEnabled(False)
        if hasattr(self, "btn_api_refresh"):
            self.btn_api_refresh.setEnabled(False)

        self.auth_check_thread = AuthCheckThread(pending)
        self.auth_check_thread.completed.connect(self.on_auth_check_completed)
        self.auth_check_thread.start()

    def reject(self):
        if self.apply_thread and self.apply_thread.isRunning():
            return
        if self.ollama_context_thread and self.ollama_context_thread.isRunning():
            return
        if self.ollama_tools_thread and self.ollama_tools_thread.isRunning():
            return
        super().reject()

    def closeEvent(self, event):
        if self.apply_thread and self.apply_thread.isRunning():
            event.ignore()
            return
        if self.ollama_context_thread and self.ollama_context_thread.isRunning():
            event.ignore()
            return
        if self.ollama_tools_thread and self.ollama_tools_thread.isRunning():
            event.ignore()
            return
        super().closeEvent(event)


class OllamaModelManagerDialog(QDialog):
    def __init__(self, parent=None, base_url: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Ollama 모델 관리")
        self.setModal(True)
        self.setMinimumSize(820, 620)
        self.resize(920, 700)
        self.setSizeGripEnabled(True)
        self.task_thread: Optional[OllamaModelTaskThread] = None
        self.base_url_default = str(base_url or "").strip() or "http://127.0.0.1:11434/v1"
        self.setStyleSheet("""
            QDialog {
                background: #0f141b;
                color: #e5e7eb;
            }
            QLabel {
                color: #e5e7eb;
            }
            QLineEdit, QComboBox, QPlainTextEdit {
                background: #0b0f14;
                color: #e5e7eb;
                border: 1px solid #2a2f3a;
                border-radius: 6px;
                padding: 4px 6px;
                min-height: 28px;
            }
            QPushButton {
                background: #1f2937;
                color: #e5e7eb;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 5px 10px;
                min-height: 28px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #374151;
            }
            QPushButton:disabled {
                color: #7a8390;
                border-color: #2d3442;
                background: #111827;
            }
        """)
        self.init_ui()
        self.refresh_cli_hint()
        self.refresh_gguf_files()
        self.start_task("list")

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        row_base = QHBoxLayout()
        row_base.addWidget(QLabel("Ollama Base URL"))
        self.base_url_input = QLineEdit(self.base_url_default)
        row_base.addWidget(self.base_url_input, 1)
        self.btn_refresh_models = QPushButton("목록 새로고침")
        self.btn_refresh_models.clicked.connect(lambda: self.start_task("list"))
        row_base.addWidget(self.btn_refresh_models)
        layout.addLayout(row_base)

        row_cli = QHBoxLayout()
        row_cli.addWidget(QLabel("Ollama CLI"))
        self.cli_hint = QLabel("")
        self.cli_hint.setStyleSheet("color: #9ca3af; font-size: 11px;")
        row_cli.addWidget(self.cli_hint, 1)
        self.btn_refresh_cli = QPushButton("CLI 재탐색")
        self.btn_refresh_cli.clicked.connect(self.refresh_cli_hint)
        row_cli.addWidget(self.btn_refresh_cli)
        layout.addLayout(row_cli)

        row_folder = QHBoxLayout()
        row_folder.addWidget(QLabel("GGUF 폴더"))
        self.folder_input = QLineEdit("")
        self.folder_input.setPlaceholderText("GGUF 파일이 있는 폴더")
        row_folder.addWidget(self.folder_input, 1)
        self.btn_browse_folder = QPushButton("찾기")
        self.btn_browse_folder.clicked.connect(self.browse_gguf_folder)
        row_folder.addWidget(self.btn_browse_folder)
        self.btn_scan_folder = QPushButton("스캔")
        self.btn_scan_folder.clicked.connect(self.refresh_gguf_files)
        row_folder.addWidget(self.btn_scan_folder)
        layout.addLayout(row_folder)

        row_gguf = QHBoxLayout()
        row_gguf.addWidget(QLabel("GGUF 파일"))
        self.gguf_combo = QComboBox()
        self.gguf_combo.currentTextChanged.connect(self.on_gguf_selected)
        row_gguf.addWidget(self.gguf_combo, 1)
        layout.addLayout(row_gguf)

        row_create = QHBoxLayout()
        row_create.addWidget(QLabel("등록 이름"))
        self.model_name_input = QLineEdit("")
        self.model_name_input.setPlaceholderText("예: deepseek-r1-32b-uncensored-q4km")
        row_create.addWidget(self.model_name_input, 1)
        row_create.addWidget(QLabel("Temp"))
        self.temperature_input = QLineEdit("0.7")
        self.temperature_input.setFixedWidth(72)
        row_create.addWidget(self.temperature_input)
        self.btn_modelfile = QPushButton("Modelfile 생성")
        self.btn_modelfile.clicked.connect(self.create_modelfile_only)
        row_create.addWidget(self.btn_modelfile)
        self.btn_create_model = QPushButton("등록(create)")
        self.btn_create_model.clicked.connect(self.create_ollama_model)
        row_create.addWidget(self.btn_create_model)
        layout.addLayout(row_create)

        row_registered = QHBoxLayout()
        row_registered.addWidget(QLabel("등록 모델"))
        self.registered_models_combo = QComboBox()
        self.registered_models_combo.setEnabled(False)
        self.registered_models_combo.addItem("(모델 없음)")
        row_registered.addWidget(self.registered_models_combo, 1)
        self.btn_remove_model = QPushButton("선택 제거")
        self.btn_remove_model.clicked.connect(self.remove_selected_model)
        row_registered.addWidget(self.btn_remove_model)
        layout.addLayout(row_registered)

        self.status_hint = QLabel("gguf를 선택해 Modelfile 생성 또는 등록(create)을 실행하세요.")
        self.status_hint.setStyleSheet("color: #9ca3af; font-size: 11px;")
        layout.addWidget(self.status_hint)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("작업 로그")
        self.log_output.setMinimumHeight(260)
        layout.addWidget(self.log_output, 1)

        row_footer = QHBoxLayout()
        row_footer.addStretch()
        btn_close = QPushButton("닫기")
        btn_close.clicked.connect(self.accept)
        row_footer.addWidget(btn_close)
        layout.addLayout(row_footer)

    def append_log(self, message: str):
        now_text = time.strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{now_text}] {message}")

    def set_busy(self, busy: bool):
        enabled = not bool(busy)
        for widget in (
            self.btn_refresh_models,
            self.btn_refresh_cli,
            self.btn_browse_folder,
            self.btn_scan_folder,
            self.btn_modelfile,
            self.btn_create_model,
            self.btn_remove_model,
            self.base_url_input,
            self.folder_input,
            self.gguf_combo,
            self.model_name_input,
            self.temperature_input,
        ):
            widget.setEnabled(enabled)

    def refresh_cli_hint(self):
        cli = resolve_ollama_cli_path()
        if cli:
            self.cli_hint.setText(cli)
            self.cli_hint.setStyleSheet("color: #22c55e; font-size: 11px;")
        else:
            self.cli_hint.setText("미탐지: OLLAMA_CLI 또는 PATH 확인")
            self.cli_hint.setStyleSheet("color: #ef4444; font-size: 11px;")

    def browse_gguf_folder(self):
        path = QFileDialog.getExistingDirectory(self, "GGUF 폴더 선택")
        if not path:
            return
        self.folder_input.setText(path)
        self.refresh_gguf_files()

    def refresh_gguf_files(self):
        folder_raw = str(self.folder_input.text() or "").strip()
        self.gguf_combo.blockSignals(True)
        self.gguf_combo.clear()
        folder_path = Path(folder_raw) if folder_raw else None
        files: list[str] = []
        if folder_path and folder_path.exists() and folder_path.is_dir():
            for item in sorted(folder_path.iterdir(), key=lambda p: p.name.lower()):
                if item.is_file() and item.suffix.lower() == ".gguf":
                    files.append(item.name)
        if files:
            self.gguf_combo.addItems(files)
            self.gguf_combo.setEnabled(True)
        else:
            self.gguf_combo.addItem("(gguf 없음)")
            self.gguf_combo.setEnabled(False)
        self.gguf_combo.blockSignals(False)
        if files:
            self.on_gguf_selected(files[0])
            self.append_log(f"GGUF 파일 {len(files)}개 감지")
        else:
            self.append_log("GGUF 파일을 찾지 못했습니다.")

    def on_gguf_selected(self, selected_name: str):
        gguf_name = str(selected_name or "").strip()
        if not gguf_name or gguf_name == "(gguf 없음)":
            return
        if not self.model_name_input.text().strip():
            auto_name = sanitize_ollama_model_name(Path(gguf_name).stem)
            if auto_name:
                self.model_name_input.setText(auto_name)

    def create_modelfile_only(self):
        folder_raw = str(self.folder_input.text() or "").strip()
        gguf_name = str(self.gguf_combo.currentText() or "").strip()
        if not folder_raw:
            QMessageBox.warning(self, "입력 필요", "GGUF 폴더를 선택하세요.")
            return
        if not gguf_name or gguf_name == "(gguf 없음)":
            QMessageBox.warning(self, "입력 필요", "GGUF 파일을 선택하세요.")
            return
        folder_path = Path(folder_raw)
        gguf_path = folder_path / gguf_name
        if not gguf_path.exists():
            QMessageBox.warning(self, "파일 없음", f"GGUF 파일이 없습니다:\n{gguf_path}")
            return
        modelfile_text = build_modelfile_text(gguf_name, self.temperature_input.text())
        modelfile_path = folder_path / "Modelfile"
        modelfile_path.write_text(modelfile_text, encoding="utf-8")
        self.append_log(f"Modelfile 생성 완료: {modelfile_path}")
        self.status_hint.setText("Modelfile 생성 완료")
        self.status_hint.setStyleSheet("color: #22c55e; font-size: 11px;")

    def set_registered_models(self, models: list[str], selected: str = ""):
        clean = [str(item).strip() for item in models if str(item).strip()]
        self.registered_models_combo.blockSignals(True)
        self.registered_models_combo.clear()
        if clean:
            self.registered_models_combo.addItems(clean)
            self.registered_models_combo.setEnabled(True)
            if selected and selected in clean:
                self.registered_models_combo.setCurrentText(selected)
            elif clean:
                self.registered_models_combo.setCurrentIndex(0)
        else:
            self.registered_models_combo.addItem("(모델 없음)")
            self.registered_models_combo.setEnabled(False)
        self.registered_models_combo.blockSignals(False)

    def start_task(self, action: str, remove_name: str = ""):
        if self.task_thread and self.task_thread.isRunning():
            return
        folder_raw = str(self.folder_input.text() or "").strip()
        gguf_name = str(self.gguf_combo.currentText() or "").strip()
        model_name = sanitize_ollama_model_name(self.model_name_input.text())
        if action == "create":
            if not folder_raw or not gguf_name or gguf_name == "(gguf 없음)":
                QMessageBox.warning(self, "입력 필요", "GGUF 폴더/파일을 확인하세요.")
                return
            if not model_name:
                QMessageBox.warning(self, "입력 필요", "등록 모델 이름을 입력하세요.")
                return
            self.model_name_input.setText(model_name)
        base_url = str(self.base_url_input.text() or "").strip() or "http://127.0.0.1:11434/v1"
        self.set_busy(True)
        self.status_hint.setText(f"{action} 작업 실행 중...")
        self.status_hint.setStyleSheet("color: #f59e0b; font-size: 11px;")
        self.task_thread = OllamaModelTaskThread(
            action=action,
            base_url=base_url,
            folder=folder_raw,
            gguf_file=gguf_name,
            model_name=model_name,
            temperature=str(self.temperature_input.text() or "0.7").strip(),
            remove_name=remove_name,
        )
        self.task_thread.completed.connect(self.on_task_completed)
        self.task_thread.start()

    def create_ollama_model(self):
        self.start_task("create")

    def remove_selected_model(self):
        model_name = str(self.registered_models_combo.currentText() or "").strip()
        if not model_name or model_name == "(모델 없음)":
            return
        confirmed = QMessageBox.question(
            self,
            "모델 제거 확인",
            f"다음 모델을 제거할까요?\n{model_name}",
        )
        if confirmed != QMessageBox.Yes:
            return
        self.start_task("remove", remove_name=model_name)

    def on_task_completed(self, action: str, ok: bool, message: str, payload_obj: object):
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        self.set_busy(False)
        self.task_thread = None
        if ok:
            models_obj = payload.get("models") if isinstance(payload, dict) else payload_obj
            models = models_obj if isinstance(models_obj, list) else []
            if action == "list":
                models = payload_obj if isinstance(payload_obj, list) else models
            self.set_registered_models(models)
            selected = str(payload.get("model") or "").strip() if isinstance(payload, dict) else ""
            if selected and selected in models:
                self.registered_models_combo.setCurrentText(selected)
            self.status_hint.setText(message)
            self.status_hint.setStyleSheet("color: #22c55e; font-size: 11px;")
            self.append_log(message)
            stdout_text = str(payload.get("stdout") or "").strip() if isinstance(payload, dict) else ""
            if stdout_text:
                if len(stdout_text) > 1600:
                    stdout_text = stdout_text[-1600:]
                self.append_log(stdout_text)
            return
        self.status_hint.setText(message)
        self.status_hint.setStyleSheet("color: #ef4444; font-size: 11px;")
        self.append_log(f"{action} 실패: {message}")
        QMessageBox.warning(self, "작업 실패", message)

    def reject(self):
        if self.task_thread and self.task_thread.isRunning():
            return
        super().reject()

    def closeEvent(self, event):
        if self.task_thread and self.task_thread.isRunning():
            event.ignore()
            return
        super().closeEvent(event)


class GhostChatLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ghost_chat_windows: list = []
        self.ghost_chat_counter = 0
        self.dashboard_process = None
        self.ollama_manager_dialog: Optional[OllamaModelManagerDialog] = None
        self.gateway_action_thread: Optional[GatewayActionThread] = None
        self.gateway_action_pending = False
        self.gateway_pending_action_label = ""
        self._last_gateway_running: Optional[bool] = None
        self.init_ui()
        self.init_tray()
        self.drag_position = QPoint()
        self.gateway_status_timer = QTimer(self)
        self.gateway_status_timer.setInterval(1500)
        self.gateway_status_timer.timeout.connect(self.refresh_gateway_status)
        self.gateway_status_timer.start()
        self.refresh_gateway_status()
        
    def init_ui(self):
        self.setWindowTitle("GhostRelay Console")
        self.setFixedSize(300, 382)
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 중앙 위젯
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 헤더 (드래그 영역 + 버튼)
        header = QWidget()
        header.setFixedHeight(35)
        header.setStyleSheet("""
            QWidget {
                background: rgba(21, 23, 28, 0.95);
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 5, 5)
        header_layout.setSpacing(5)
        
        title = QLabel("GhostRelay Console")
        title.setStyleSheet("color: white; font-weight: 600; font-size: 13px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # 트레이 버튼
        btn_tray = QPushButton("🗕")
        btn_tray.setFixedSize(24, 24)
        btn_tray.setToolTip("트레이로 최소화")
        btn_tray.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.1);
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.2);
            }
        """)
        btn_tray.clicked.connect(self.hide_to_tray)
        header_layout.addWidget(btn_tray)
        
        # 닫기 버튼
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(24, 24)
        btn_close.setToolTip("닫기")
        btn_close.setStyleSheet("""
            QPushButton {
                background: rgba(255, 123, 123, 0.2);
                color: #ff7b7b;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(255, 123, 123, 0.4);
            }
        """)
        btn_close.clicked.connect(self.close_app)
        header_layout.addWidget(btn_close)
        
        layout.addWidget(header)
        
        # 버튼 영역
        buttons_widget = QWidget()
        buttons_layout = QVBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(10, 10, 10, 10)
        buttons_layout.setSpacing(8)
        
        # 스킬 대시보드 버튼
        btn_dashboard = QPushButton("📊 스킬 대시보드")
        btn_dashboard.setStyleSheet("""
            QPushButton {
                background: #6bd0ff;
                color: #071018;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #5bb8e0;
            }
        """)
        btn_dashboard.clicked.connect(self.open_dashboard)
        buttons_layout.addWidget(btn_dashboard)
        
        # 고스트 챗 버튼
        btn_ghost = QPushButton("👻 브레인 채팅")
        btn_ghost.setStyleSheet("""
            QPushButton {
                background: #86efac;
                color: #071018;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #6ee7b7;
            }
        """)
        btn_ghost.clicked.connect(self.open_ghost_chat)
        buttons_layout.addWidget(btn_ghost)

        btn_companion = QPushButton("💬 대화 전용 채팅")
        btn_companion.setStyleSheet("""
            QPushButton {
                background: #c4b5fd;
                color: #101118;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #a78bfa;
            }
        """)
        btn_companion.clicked.connect(self.open_companion_chat)
        buttons_layout.addWidget(btn_companion)

        btn_ollama_manager = QPushButton("🧱 Ollama 모델 관리")
        btn_ollama_manager.setStyleSheet("""
            QPushButton {
                background: #facc15;
                color: #151515;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #eab308;
            }
        """)
        btn_ollama_manager.clicked.connect(self.open_ollama_model_manager)
        buttons_layout.addWidget(btn_ollama_manager)

        gateway_status_row = QHBoxLayout()
        gateway_status_row.setContentsMargins(0, 2, 0, 2)
        gateway_status_row.setSpacing(6)

        self.gateway_status_dot = QLabel("●")
        self.gateway_status_dot.setStyleSheet("color: #ef4444; font-size: 13px;")
        gateway_status_row.addWidget(self.gateway_status_dot)

        self.gateway_status_text = QLabel("Gateway: 오프라인")
        self.gateway_status_text.setStyleSheet("color: #e5e7eb; font-size: 12px;")
        gateway_status_row.addWidget(self.gateway_status_text, 1)
        buttons_layout.addLayout(gateway_status_row)

        gateway_buttons_row = QHBoxLayout()
        gateway_buttons_row.setContentsMargins(0, 0, 0, 0)
        gateway_buttons_row.setSpacing(6)

        self.btn_gateway_start = QPushButton("게이트 시작")
        self.btn_gateway_start.setObjectName("gatewayStart")
        self.btn_gateway_start.clicked.connect(lambda: self.run_gateway_action("start"))
        gateway_buttons_row.addWidget(self.btn_gateway_start)

        self.btn_gateway_stop = QPushButton("게이트 중지")
        self.btn_gateway_stop.setObjectName("gatewayStop")
        self.btn_gateway_stop.clicked.connect(lambda: self.run_gateway_action("stop"))
        gateway_buttons_row.addWidget(self.btn_gateway_stop)

        self.btn_gateway_restart = QPushButton("재시작")
        self.btn_gateway_restart.setObjectName("gatewayRestart")
        self.btn_gateway_restart.clicked.connect(lambda: self.run_gateway_action("restart"))
        gateway_buttons_row.addWidget(self.btn_gateway_restart)

        buttons_layout.addLayout(gateway_buttons_row)
        self.gateway_action_hint = QLabel("게이트웨이는 콘솔에서 관리됩니다.")
        self.gateway_action_hint.setStyleSheet("color: #9ca3af; font-size: 11px;")
        buttons_layout.addWidget(self.gateway_action_hint)

        buttons_widget.setStyleSheet("""
            QWidget {
                background: rgba(21, 23, 28, 0.9);
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }
            QPushButton {
                background: #1f2937;
                color: #e5e7eb;
                border: 1px solid #374151;
                padding: 7px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #374151;
            }
            QPushButton:disabled {
                color: #7a8390;
                border-color: #2d3442;
                background: #111827;
            }
            QPushButton#gatewayStart {
                background: #123a28;
                border: 1px solid #1f6a45;
                color: #b7f7d4;
            }
            QPushButton#gatewayStop {
                background: #3a1a1a;
                border: 1px solid #7f1d1d;
                color: #fecaca;
            }
            QPushButton#gatewayRestart {
                background: #3a2f12;
                border: 1px solid #8a6a16;
                color: #fde68a;
            }
            QPushButton#gatewayStart:hover {
                background: #185137;
            }
            QPushButton#gatewayStop:hover {
                background: #5a2323;
            }
            QPushButton#gatewayRestart:hover {
                background: #57471b;
            }
        """)
        
        layout.addWidget(buttons_widget)
        
    def init_tray(self):
        """시스템 트레이 아이콘 설정"""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
            
        self.tray = QSystemTrayIcon(self)
        try:
            self.tray.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        except Exception:
            pass
        self.tray.setToolTip("GhostRelay Console")
        self.tray_menu = QMenu()
        self.tray_menu.aboutToShow.connect(self.rebuild_tray_menu)
        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(self.tray_activated)
        self.rebuild_tray_menu()
        # 아이콘 없이도 트레이에 표시 (경고는 무시)
        try:
            self.tray.show()
        except:
            pass  # 아이콘 없어도 작동하도록

    def get_alive_ghost_windows(self) -> list:
        alive = []
        for window in self.ghost_chat_windows:
            try:
                if window is not None:
                    window.isVisible()
                    alive.append(window)
            except RuntimeError:
                continue
            except Exception:
                continue
        self.ghost_chat_windows = alive
        return alive

    def describe_ghost_window(self, window, index: int) -> str:
        try:
            profile = str(getattr(window, "profile_id", "") or "").strip() or "main"
        except Exception:
            profile = "main"
        try:
            seq = int(window.property("ghostchat_seq") or index)
        except Exception:
            seq = index
        try:
            visible = bool(window.isVisible())
        except Exception:
            visible = False
        state = "열림" if visible else "숨김"
        mode_raw = str(window.property("ghostchat_mode") or "").strip().lower()
        if mode_raw == "companion":
            mode_label = "대화"
        else:
            mode_label = "브레인"
        return f"{seq}. [{profile}/{mode_label}] {state}"

    def focus_ghost_chat(self, window):
        try:
            window.showNormal()
        except Exception:
            try:
                window.show()
            except Exception:
                return
        try:
            window.raise_()
            window.activateWindow()
        except Exception:
            pass

    def open_ghost_chat_settings(self, window):
        self.focus_ghost_chat(window)
        try:
            window.open_embedded_settings()
        except Exception:
            pass

    def hide_ghost_chat_window(self, window):
        try:
            window.hide()
        except Exception:
            pass

    def close_ghost_chat_window(self, window):
        try:
            if hasattr(window, "quit_window"):
                window.quit_window()
            else:
                if hasattr(window, "_force_close"):
                    window._force_close = True
                window.close()
        except Exception:
            pass
        self.cleanup_closed_ghost_windows()

    def rebuild_tray_menu(self):
        if not hasattr(self, "tray_menu") or self.tray_menu is None:
            return
        menu = self.tray_menu
        menu.clear()

        show_action = QAction("보이기", self)
        show_action.triggered.connect(self.show)
        menu.addAction(show_action)

        hide_action = QAction("숨기기", self)
        hide_action.triggered.connect(self.hide)
        menu.addAction(hide_action)

        menu.addSeparator()

        dashboard_action = QAction("📊 Skills Dashboard", self)
        dashboard_action.triggered.connect(self.open_dashboard)
        menu.addAction(dashboard_action)

        new_ghost_action = QAction("👻 Ghost Chat 새 창", self)
        new_ghost_action.triggered.connect(self.open_ghost_chat)
        menu.addAction(new_ghost_action)

        new_companion_action = QAction("💬 대화 전용 채팅 새 창", self)
        new_companion_action.triggered.connect(self.open_companion_chat)
        menu.addAction(new_companion_action)

        ollama_manage_action = QAction("🧱 Ollama 모델 관리", self)
        ollama_manage_action.triggered.connect(self.open_ollama_model_manager)
        menu.addAction(ollama_manage_action)

        windows = self.get_alive_ghost_windows()
        menu.addSeparator()
        windows_header = QAction(f"Ghost Chat 창 {len(windows)}개", self)
        windows_header.setEnabled(False)
        menu.addAction(windows_header)

        if not windows:
            empty_action = QAction("실행 중인 Ghost Chat 없음", self)
            empty_action.setEnabled(False)
            menu.addAction(empty_action)
        else:
            for idx, window in enumerate(windows, start=1):
                window_menu = menu.addMenu(self.describe_ghost_window(window, idx))

                action_show = QAction("열기/포커스", self)
                action_show.triggered.connect(lambda _=False, w=window: self.focus_ghost_chat(w))
                window_menu.addAction(action_show)

                action_settings = QAction("설정", self)
                action_settings.triggered.connect(lambda _=False, w=window: self.open_ghost_chat_settings(w))
                window_menu.addAction(action_settings)

                action_hide = QAction("트레이로 숨기기", self)
                action_hide.triggered.connect(lambda _=False, w=window: self.hide_ghost_chat_window(w))
                window_menu.addAction(action_hide)

                action_close = QAction("종료", self)
                action_close.triggered.connect(lambda _=False, w=window: self.close_ghost_chat_window(w))
                window_menu.addAction(action_close)

        menu.addSeparator()

        gateway_state = "온라인" if check_gateway_running() else "오프라인"
        gateway_header = QAction(f"Gateway: {gateway_state}", self)
        gateway_header.setEnabled(False)
        menu.addAction(gateway_header)

        start_gateway_action = QAction("게이트웨이 시작", self)
        start_gateway_action.triggered.connect(lambda: self.run_gateway_action("start"))
        menu.addAction(start_gateway_action)

        stop_gateway_action = QAction("게이트웨이 중지", self)
        stop_gateway_action.triggered.connect(lambda: self.run_gateway_action("stop"))
        menu.addAction(stop_gateway_action)

        restart_gateway_action = QAction("게이트웨이 재시작", self)
        restart_gateway_action.triggered.connect(lambda: self.run_gateway_action("restart"))
        menu.addAction(restart_gateway_action)

        menu.addSeparator()

        quit_action = QAction("종료", self)
        quit_action.triggered.connect(self.close_app)
        menu.addAction(quit_action)
        
    def tray_activated(self, reason):
        """트레이 아이콘 클릭 시"""
        if reason == QSystemTrayIcon.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                
    def hide_to_tray(self):
        """트레이로 최소화"""
        if self.tray.isVisible():
            self.hide()
        
    def close_app(self):
        """앱 종료"""
        if self.gateway_action_thread and self.gateway_action_thread.isRunning():
            self.gateway_action_thread.terminate()
            self.gateway_action_thread.wait(1000)
        if self.dashboard_process and self.dashboard_process.poll() is None:
            self.dashboard_process.terminate()
            try:
                self.dashboard_process.wait(timeout=3)
            except:
                self.dashboard_process.kill()
        for window in list(self.ghost_chat_windows):
            self.close_ghost_chat_window(window)
        self.ghost_chat_windows.clear()
        try:
            if hasattr(self, "tray") and self.tray:
                self.tray.hide()
        except Exception:
            pass
        QApplication.quit()
        
    def open_dashboard(self):
        """스킬 대시보드 열기"""
        ghostchat_dir = Path(__file__).parent
        dashboard_dir = ghostchat_dir / "dashboard"
        start_bat = dashboard_dir / "start.bat"
        
        if start_bat.exists():
            if self.dashboard_process is None or self.dashboard_process.poll() is not None:
                self.dashboard_process = subprocess.Popen(
                    ["cmd", "/c", str(start_bat)],
                    cwd=str(dashboard_dir),
                    **hidden_popen_kwargs(),
                )
        else:
            webbrowser.open("http://127.0.0.1:5177")
        self.hide_to_tray()
            
    def find_ghost_window(self, profile_id: str, mode: str):
        target_profile = normalize_profile_id(profile_id)
        target_mode = str(mode or "brain").strip().lower()
        if target_mode not in {"brain", "companion"}:
            target_mode = "brain"
        for window in self.get_alive_ghost_windows():
            try:
                current_profile = normalize_profile_id(str(getattr(window, "profile_id", "") or "main"))
            except Exception:
                current_profile = "main"
            try:
                current_mode = str(window.property("ghostchat_mode") or "brain").strip().lower()
            except Exception:
                current_mode = "brain"
            if current_mode not in {"brain", "companion"}:
                current_mode = "brain"
            if current_profile == target_profile and current_mode == target_mode:
                return window
        return None

    def ensure_brain_window_visible(self, profile_id: str, focus_if_hidden: bool = True):
        target_profile = normalize_profile_id(profile_id)
        existing = self.find_ghost_window(target_profile, "brain")
        if existing is not None:
            try:
                is_visible = bool(existing.isVisible())
            except Exception:
                is_visible = False
            if not is_visible:
                try:
                    existing.showNormal()
                except Exception:
                    try:
                        existing.show()
                    except Exception:
                        pass
                if focus_if_hidden:
                    self.focus_ghost_chat(existing)
            return existing
        return self.open_ghost_chat(
            profile_id=target_profile,
            reuse_existing=False,
            hide_console=False,
        )

    def on_companion_brain_monitor_requested(self, profile_id: str, reason: str):
        _ = reason
        target_profile = normalize_profile_id(profile_id or get_active_profile_id(load_settings()))
        self.ensure_brain_window_visible(target_profile, focus_if_hidden=True)
        self.rebuild_tray_menu()

    def on_companion_brain_monitor_log(self, profile_id: str, text: str):
        target_profile = normalize_profile_id(profile_id or get_active_profile_id(load_settings()))
        target_window = self.find_ghost_window(target_profile, "brain")
        if target_window is None:
            target_window = self.ensure_brain_window_visible(target_profile, focus_if_hidden=False)
        if target_window is None:
            return
        message = str(text or "").strip()
        if not message:
            return
        try:
            target_window.append_system_line(f"ℹ️ {message}")
        except Exception:
            pass

    def open_ghost_chat(
        self,
        profile_id: Optional[str] = None,
        reuse_existing: bool = False,
        hide_console: bool = True,
    ):
        """고스트 챗 열기"""
        from ghost_chat import GhostChatWindow
        settings = load_settings()
        target_profile = normalize_profile_id(profile_id or get_active_profile_id(settings))
        if reuse_existing:
            existing = self.find_ghost_window(target_profile, "brain")
            if existing is not None:
                self.focus_ghost_chat(existing)
                if hide_console:
                    self.hide_to_tray()
                return existing
        window = GhostChatWindow(
            profile_id=target_profile,
            use_tray_icon=False,
            console_managed=True,
        )
        self.ghost_chat_counter += 1
        try:
            window.setProperty("ghostchat_seq", self.ghost_chat_counter)
            window.setProperty("ghostchat_mode", "brain")
        except Exception:
            pass
        window.show()
        self.ghost_chat_windows.append(window)
        window.destroyed.connect(self.cleanup_closed_ghost_windows)
        self.rebuild_tray_menu()
        if hide_console:
            self.hide_to_tray()
        return window

    def open_companion_chat(
        self,
        profile_id: Optional[str] = None,
        hide_console: bool = True,
        ensure_brain_monitor: bool = True,
    ):
        """대화 전용 채팅 열기 (브레인 브리지 포함)"""
        from ghost_chat_companion import GhostChatCompanionWindow
        settings = load_settings()
        target_profile = normalize_profile_id(profile_id or get_active_profile_id(settings))
        window = GhostChatCompanionWindow(
            profile_id=target_profile,
            use_tray_icon=False,
            console_managed=True,
        )
        try:
            window.brain_monitor_requested.connect(self.on_companion_brain_monitor_requested)
            window.brain_monitor_log.connect(self.on_companion_brain_monitor_log)
        except Exception:
            pass
        self.ghost_chat_counter += 1
        try:
            window.setProperty("ghostchat_seq", self.ghost_chat_counter)
            window.setProperty("ghostchat_mode", "companion")
        except Exception:
            pass
        window.show()
        self.ghost_chat_windows.append(window)
        window.destroyed.connect(self.cleanup_closed_ghost_windows)
        if ensure_brain_monitor:
            try:
                bridge = window.get_brain_bridge_config()
                brain_profile = normalize_profile_id(
                    str(bridge.get("profileId") or target_profile)
                )
            except Exception:
                brain_profile = target_profile
            self.ensure_brain_window_visible(brain_profile, focus_if_hidden=True)
        self.rebuild_tray_menu()
        if hide_console:
            self.hide_to_tray()
        return window

    def open_ollama_model_manager(self):
        settings = load_settings()
        profile_id = get_active_profile_id(settings)
        profile = get_profile(settings, profile_id)
        ollama = profile.get("ollama", {}) if isinstance(profile.get("ollama"), dict) else {}
        base_url = str(ollama.get("baseUrl") or "http://127.0.0.1:11434/v1").strip()
        dialog = OllamaModelManagerDialog(self, base_url=base_url)
        self.ollama_manager_dialog = dialog
        dialog.exec()
        self.ollama_manager_dialog = None

    def cleanup_closed_ghost_windows(self):
        self.get_alive_ghost_windows()
        if hasattr(self, "tray_menu") and self.tray_menu is not None:
            self.rebuild_tray_menu()

    def get_active_profile_settings(self) -> dict:
        settings = load_settings()
        profile_id = get_active_profile_id(settings)
        return get_profile(settings, profile_id)

    def set_gateway_controls_enabled(self, enabled: bool):
        self.btn_gateway_start.setEnabled(enabled)
        self.btn_gateway_stop.setEnabled(enabled)
        self.btn_gateway_restart.setEnabled(enabled)

    def refresh_gateway_status(self):
        if self.gateway_action_pending:
            self.gateway_status_dot.setStyleSheet("color: #f59e0b; font-size: 13px;")
            self.gateway_status_text.setText(f"Gateway: {self.gateway_pending_action_label} 중...")
            return
        running = check_gateway_running()
        if running:
            self.gateway_status_dot.setStyleSheet("color: #22c55e; font-size: 13px;")
            self.gateway_status_text.setText("Gateway: 온라인")
        else:
            self.gateway_status_dot.setStyleSheet("color: #ef4444; font-size: 13px;")
            self.gateway_status_text.setText("Gateway: 오프라인")
        desired_interval = 1500 if running else 3000
        if self.gateway_status_timer.interval() != desired_interval:
            self.gateway_status_timer.setInterval(desired_interval)
        if running != self._last_gateway_running:
            self._last_gateway_running = running
            if hasattr(self, "tray_menu") and self.tray_menu is not None and not self.tray_menu.isVisible():
                self.rebuild_tray_menu()

    def run_gateway_action(self, action: str):
        if self.gateway_action_thread and self.gateway_action_thread.isRunning():
            self.gateway_action_hint.setText("게이트웨이 작업 진행 중입니다...")
            return
        profile = self.get_active_profile_settings()
        self.gateway_action_pending = True
        self.gateway_pending_action_label = "재시작" if action == "restart" else ("시작" if action == "start" else "중지")
        self.set_gateway_controls_enabled(False)
        if action == "start":
            self.gateway_action_hint.setText("게이트웨이 시작 중...")
        elif action == "stop":
            self.gateway_action_hint.setText("게이트웨이 중지 중...")
        else:
            self.gateway_action_hint.setText("게이트웨이 재시작 중...")
        self.gateway_action_hint.setStyleSheet("color: #f59e0b; font-size: 11px;")
        self.refresh_gateway_status()
        self.gateway_action_thread = GatewayActionThread(action, profile)
        self.gateway_action_thread.progress.connect(self.on_gateway_action_progress)
        self.gateway_action_thread.completed.connect(self.on_gateway_action_completed)
        self.gateway_action_thread.start()

    def on_gateway_action_progress(self, action: str, step: int, total: int, detail: str):
        action_label = "시작" if action == "start" else ("중지" if action == "stop" else "재시작")
        now_text = time.strftime("%H:%M:%S")
        self.gateway_action_hint.setText(f"[{now_text}] {action_label} {step}/{total} - {detail}")
        self.gateway_action_hint.setStyleSheet("color: #f59e0b; font-size: 11px;")

    def on_gateway_action_completed(self, action: str, ok: bool, detail: str):
        self.gateway_action_pending = False
        self.gateway_pending_action_label = ""
        self.set_gateway_controls_enabled(True)
        self.refresh_gateway_status()
        action_label = "시작" if action == "start" else ("중지" if action == "stop" else "재시작")
        prefix = "완료" if ok else "실패"
        if not ok and check_gateway_running():
            detail = f"{detail} (기존 게이트웨이는 계속 온라인)"
        now_text = time.strftime("%H:%M:%S")
        self.gateway_action_hint.setText(f"[{now_text}] {action_label} {prefix} - {detail}")
        self.gateway_action_hint.setStyleSheet(
            "color: #22c55e; font-size: 11px;" if ok else "color: #ef4444; font-size: 11px;"
        )
        self.gateway_action_thread = None

    def mousePressEvent(self, event):
        """드래그 시작"""
        if event.button() == Qt.LeftButton:
            global_point = (
                event.globalPosition().toPoint()
                if hasattr(event, "globalPosition")
                else event.globalPos()
            )
            self.drag_position = global_point - self.frameGeometry().topLeft()
            event.accept()
            
    def mouseMoveEvent(self, event):
        """드래그 중"""
        if event.buttons() == Qt.LeftButton:
            global_point = (
                event.globalPosition().toPoint()
                if hasattr(event, "globalPosition")
                else event.globalPos()
            )
            self.move(global_point - self.drag_position)
            event.accept()
            
    def closeEvent(self, event):
        """창 닫기 시 트레이로 최소화"""
        if hasattr(self, "tray") and self.tray.isVisible():
            self.hide()
            event.ignore()
        else:
            self.close_app()
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    
    launcher = GhostChatLauncher()
    launcher.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

