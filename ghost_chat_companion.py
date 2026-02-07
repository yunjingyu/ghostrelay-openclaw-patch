"""
Ghost Chat Companion Mode
- Chat model stays tool-off
- Separate Brain model can be selected and pre-warmed in agent mode
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional
from urllib import request, error

from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QPushButton,
    QTabWidget,
)

from ghost_chat import (
    GhostChatWindow,
    OpenClawGatewayManager,
    build_agent_session_key,
    get_profile,
    list_profiles,
    load_settings,
    normalize_agent_id,
)


def strip_ollama_prefix(model_ref: str) -> str:
    raw = str(model_ref or "").strip()
    if raw.lower().startswith("ollama/"):
        return raw.split("/", 1)[1].strip()
    return raw


def is_error_like_brain_message(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    error_patterns = (
        "http 4",
        "http 5",
        "error:",
        "model not found",
        "unknown model",
        "does not support tools",
        "unauthorized",
        "resource exhausted",
        "timed out",
        "not found",
    )
    return any(pattern in text for pattern in error_patterns)


def normalize_ollama_tags_url(base_url: str) -> str:
    raw = (str(base_url or "").strip().rstrip("/")) or "http://127.0.0.1:11434/v1"
    if raw.endswith("/v1"):
        raw = raw[:-3]
    return f"{raw}/api/tags"


def normalize_ollama_openai_url(base_url: str) -> str:
    raw = (str(base_url or "").strip().rstrip("/")) or "http://127.0.0.1:11434/v1"
    if not raw.endswith("/v1"):
        raw = f"{raw}/v1"
    return f"{raw}/chat/completions"


def fetch_ollama_model_names(base_url: str) -> list[str]:
    tags_url = normalize_ollama_tags_url(base_url)
    req = request.Request(tags_url, method="GET")
    with request.urlopen(req, timeout=6) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    data = json.loads(raw or "{}")
    models_raw = data.get("models") or []
    names: list[str] = []
    if isinstance(models_raw, list):
        for item in models_raw:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                if name:
                    names.append(name)
    return names


def resolve_ollama_chat_target(profile_settings: dict, preferred_model_ref: str) -> tuple[str, str]:
    ollama = profile_settings.get("ollama", {}) if isinstance(profile_settings.get("ollama"), dict) else {}
    base_url = str(ollama.get("baseUrl") or "http://127.0.0.1:11434/v1").strip()
    preferred = strip_ollama_prefix(preferred_model_ref)
    configured = str(ollama.get("modelName") or "").strip()
    candidates = [item for item in [preferred, configured] if item]
    names = fetch_ollama_model_names(base_url)
    lower_map = {name.lower(): name for name in names}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found:
            return base_url, found
    if names:
        return base_url, names[0]
    return base_url, preferred or configured or "llama3.3"


def send_direct_ollama_chat(
    profile_settings: dict,
    model_ref: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 256,
) -> tuple[bool, str]:
    ollama = profile_settings.get("ollama", {}) if isinstance(profile_settings.get("ollama"), dict) else {}
    api_key = str(ollama.get("apiKey") or "ollama-local").strip() or "ollama-local"
    base_url, model_name = resolve_ollama_chat_target(profile_settings, model_ref)
    endpoint = normalize_ollama_openai_url(base_url)
    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": int(max_tokens),
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = str(e)
        return False, f"HTTP {e.code}: {body}"
    except Exception as e:
        return False, str(e)
    try:
        obj = json.loads(raw or "{}")
        choices = obj.get("choices") or []
        if choices and isinstance(choices, list):
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "").strip()
            if content:
                return True, content
        return False, "Ollama 응답이 비어 있습니다."
    except Exception as e:
        return False, f"Ollama 응답 파싱 실패: {e}"


def verify_ollama_model_exists(profile_settings: dict, model_ref: str) -> tuple[bool, str]:
    ollama = profile_settings.get("ollama", {}) if isinstance(profile_settings.get("ollama"), dict) else {}
    model_name = strip_ollama_prefix(model_ref)
    if not model_name:
        return False, "Ollama 모델명이 비어 있습니다."
    tags_url = normalize_ollama_tags_url(str(ollama.get("baseUrl") or ""))
    req = request.Request(tags_url, method="GET")
    try:
        with request.urlopen(req, timeout=6) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        data = json.loads(raw or "{}")
        models_raw = data.get("models") or []
        available: list[str] = []
        if isinstance(models_raw, list):
            for item in models_raw:
                if isinstance(item, dict):
                    name = str(item.get("name") or "").strip()
                    if name:
                        available.append(name)
        available_set = {name.lower() for name in available}
        if model_name.lower() in available_set:
            return True, "ok"
        sample = ", ".join(available[:6]) if available else "(empty)"
        return False, f"Ollama 모델 없음: {model_name} (available: {sample})"
    except Exception as e:
        return False, f"Ollama 모델 조회 실패: {e}"


class BrainPrepareThread(QThread):
    completed = Signal(bool, str, object)

    def __init__(
        self,
        profile_settings: dict,
        profile_id: str,
        brain_model: str,
        brain_agent_id: str,
        brain_session_key: str,
        context_summary: str,
    ):
        super().__init__()
        self.profile_settings = dict(profile_settings) if isinstance(profile_settings, dict) else {}
        self.profile_id = str(profile_id or "main").strip() or "main"
        self.brain_model = str(brain_model or "").strip()
        self.brain_agent_id = normalize_agent_id(brain_agent_id or f"{self.profile_id}-brain")
        self.brain_session_key = str(brain_session_key or "").strip()
        self.context_summary = str(context_summary or "").strip()

    def infer_model_source(self, model_ref: str, fallback: str) -> str:
        raw = str(model_ref or "").strip().lower()
        if raw.startswith("ollama/"):
            return "ollama"
        if raw.startswith("google-vertex/"):
            return "vertex"
        return fallback if fallback in {"vertex", "ollama"} else "vertex"

    def build_bridge_profile(self) -> dict:
        profile = dict(self.profile_settings)
        profile["agentId"] = self.brain_agent_id
        profile["model"] = self.brain_model
        source = self.infer_model_source(self.brain_model, str(profile.get("modelSource") or "vertex"))
        profile["modelSource"] = source
        # Companion brain warmup should work even for non-tool models.
        profile["toolMode"] = "chat"
        if source == "ollama":
            ollama = profile.get("ollama", {})
            if not isinstance(ollama, dict):
                ollama = {}
            model_name = self.brain_model.split("/", 1)[1].strip() if self.brain_model.startswith("ollama/") else self.brain_model
            ollama["modelName"] = model_name
            profile["ollama"] = ollama
        return profile

    def send_prepare_ping(self, token: str) -> tuple[bool, str]:
        payload = {
            "model": self.brain_model or "openclaw:main",
            "stream": False,
            "max_tokens": 64,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Brain worker session warmup. "
                        "Read hidden context and reply with a short Korean readiness confirmation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "세션 준비 요청입니다.\n"
                        f"프로필: {self.profile_id}\n"
                        f"브레인 모델: {self.brain_model}\n"
                        f"컨텍스트 요약:\n{self.context_summary}\n\n"
                        "작업 실행 없이 준비 완료 상태만 한 줄로 알려주세요."
                    ),
                },
            ],
        }
        req = request.Request(
            "http://127.0.0.1:18789/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "x-openclaw-session-key": self.brain_session_key,
                "x-openclaw-agent-id": self.brain_agent_id,
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=45) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
        except error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = str(e)
            return False, f"HTTP {e.code}: {body}"
        except Exception as e:
            return False, str(e)
        try:
            obj = json.loads(raw or "{}")
            choices = obj.get("choices") or []
            if choices and isinstance(choices, list):
                message = choices[0].get("message") or {}
                content = str(message.get("content") or "").strip()
                if content:
                    if is_error_like_brain_message(content):
                        return False, content
                    return True, content
            return True, "브레인 세션 준비 요청 완료"
        except Exception:
            return True, "브레인 세션 준비 요청 완료"

    def run(self):
        try:
            if not self.brain_model:
                self.completed.emit(False, "브레인 모델이 비어 있습니다.", {})
                return
            manager = OpenClawGatewayManager()
            manager.settings = self.build_bridge_profile()
            source = str(manager.settings.get("modelSource") or "").strip().lower()
            if source == "ollama":
                exists_ok, exists_msg = verify_ollama_model_exists(manager.settings, self.brain_model)
                if not exists_ok:
                    self.completed.emit(False, exists_msg, {})
                    return
                ollama_ok, ollama_msg = send_direct_ollama_chat(
                    manager.settings,
                    self.brain_model,
                    "세션 준비 확인 응답만 한 줄로 해주세요.",
                    (
                        f"프로필={self.profile_id}\n"
                        f"브레인 모델={self.brain_model}\n"
                        f"세션={self.brain_session_key}\n"
                        "준비 완료 여부만 짧게 답하세요."
                    ),
                    max_tokens=48,
                )
                if not ollama_ok:
                    self.completed.emit(False, ollama_msg, {})
                    return
                payload = {
                    "agentId": self.brain_agent_id,
                    "sessionKey": self.brain_session_key,
                    "model": self.brain_model,
                    "runtimeDetails": ["direct-ollama"],
                }
                self.completed.emit(True, ollama_msg, payload)
                return
            ok, details, _ = manager.sync_active_profile_runtime()
            sync_warning = ""
            if not ok:
                detail_text = "\n".join(str(item) for item in details[:6]) if isinstance(details, list) else "runtime sync failed"
                sync_warning = f"(sync warn) {detail_text}"
            if not manager.check_gateway_running():
                self.completed.emit(False, "게이트웨이가 오프라인입니다.", {})
                return
            token = manager.get_gateway_token()
            if not token:
                self.completed.emit(False, "게이트웨이 토큰을 찾지 못했습니다.", {})
                return
            ping_ok, ping_msg = self.send_prepare_ping(token)
            if not ping_ok:
                if "does not support tools" in str(ping_msg or "").lower():
                    fallback_ok, fallback_msg = send_direct_ollama_chat(
                        manager.settings,
                        self.brain_model,
                        "세션 준비 확인 응답만 한 줄로 해주세요.",
                        (
                            f"프로필={self.profile_id}\n"
                            f"브레인 모델={self.brain_model}\n"
                            f"세션={self.brain_session_key}\n"
                            "준비 완료 여부만 짧게 답하세요."
                        ),
                        max_tokens=48,
                    )
                    if fallback_ok:
                        payload = {
                            "agentId": self.brain_agent_id,
                            "sessionKey": self.brain_session_key,
                            "model": self.brain_model,
                            "runtimeDetails": ["fallback-direct-ollama"],
                        }
                        final_msg = f"{fallback_msg} {sync_warning}".strip()
                        self.completed.emit(True, final_msg, payload)
                        return
                self.completed.emit(False, ping_msg, {})
                return
            payload = {
                "agentId": self.brain_agent_id,
                "sessionKey": self.brain_session_key,
                "model": self.brain_model,
                "runtimeDetails": details,
            }
            final_msg = f"{ping_msg} {sync_warning}".strip()
            self.completed.emit(True, final_msg, payload)
        except Exception as e:
            self.completed.emit(False, str(e), {})


class BrainMemorySyncThread(QThread):
    completed = Signal(bool, str, object)

    def __init__(
        self,
        profile_settings: dict,
        profile_id: str,
        brain_model: str,
        brain_agent_id: str,
        brain_session_key: str,
        daily_memory_entry: str,
        daily_memory_path: str,
        summary_file_path: str,
    ):
        super().__init__()
        self.profile_settings = dict(profile_settings) if isinstance(profile_settings, dict) else {}
        self.profile_id = str(profile_id or "main").strip() or "main"
        self.brain_model = str(brain_model or "").strip()
        self.brain_agent_id = normalize_agent_id(brain_agent_id or f"{self.profile_id}-brain")
        self.brain_session_key = str(brain_session_key or "").strip()
        self.daily_memory_entry = str(daily_memory_entry or "").strip()
        self.daily_memory_path = str(daily_memory_path or "").strip()
        self.summary_file_path = str(summary_file_path or "").strip()

    def infer_model_source(self, model_ref: str, fallback: str) -> str:
        raw = str(model_ref or "").strip().lower()
        if raw.startswith("ollama/"):
            return "ollama"
        if raw.startswith("google-vertex/"):
            return "vertex"
        return fallback if fallback in {"vertex", "ollama"} else "vertex"

    def build_bridge_profile(self) -> dict:
        profile = dict(self.profile_settings)
        profile["agentId"] = self.brain_agent_id
        profile["model"] = self.brain_model
        source = self.infer_model_source(self.brain_model, str(profile.get("modelSource") or "vertex"))
        profile["modelSource"] = source
        # Memory sync in Companion mode is summarization-only, no tools required.
        profile["toolMode"] = "chat"
        if source == "ollama":
            ollama = profile.get("ollama", {})
            if not isinstance(ollama, dict):
                ollama = {}
            model_name = self.brain_model.split("/", 1)[1].strip() if self.brain_model.startswith("ollama/") else self.brain_model
            ollama["modelName"] = model_name
            profile["ollama"] = ollama
        return profile

    def send_sync_request(self, token: str) -> tuple[bool, str]:
        payload = {
            "model": self.brain_model or "openclaw:main",
            "stream": False,
            "max_tokens": 220,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "당신은 대화 메모 정리 요약기입니다. "
                        "도구 호출이나 코드블록 없이 순수 텍스트로만 답하세요. "
                        "핵심 사실 위주로 짧고 명확하게 요약하세요."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "새 Daily Memory 엔트리가 추가되었습니다. 아래 델타를 읽고 "
                        "장기기억 주입용 요약을 만들어 주세요.\n\n"
                        f"[profile]\n{self.profile_id}\n\n"
                        f"[source_daily_file]\n{self.daily_memory_path}\n\n"
                        "[latest_daily_memory_entry]\n"
                        f"{self.daily_memory_entry}\n\n"
                        "출력 형식(마크다운):\n"
                        "## 핵심 사실\n"
                        "- ...\n"
                        "## 진행/결정\n"
                        "- ...\n"
                        "## 다음 주입 추천\n"
                        "- ...\n"
                        "제약: 12줄 내외, 중복/군더더기 금지."
                    ),
                },
            ],
        }
        req = request.Request(
            "http://127.0.0.1:18789/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "x-openclaw-session-key": self.brain_session_key,
                "x-openclaw-agent-id": self.brain_agent_id,
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
        except error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = str(e)
            return False, f"HTTP {e.code}: {body}"
        except Exception as e:
            return False, str(e)

        try:
            obj = json.loads(raw or "{}")
            choices = obj.get("choices") or []
            if choices and isinstance(choices, list):
                message = choices[0].get("message") or {}
                content = str(message.get("content") or "").strip()
                if content:
                    if is_error_like_brain_message(content):
                        return False, content
                    return True, content
            return False, "브레인 요약 응답이 비어 있습니다."
        except Exception:
            return False, "브레인 요약 응답 파싱 실패"

    def append_summary_file(self, summary_text: str) -> tuple[bool, str]:
        target_raw = self.summary_file_path or ""
        if not target_raw:
            return False, "요약 파일 경로가 비어 있습니다."
        target = Path(target_raw)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text(
                    "# Brain Summary\n\n"
                    "- Daily 메모 델타 기반 자동 요약 로그\n\n",
                    encoding="utf-8",
                )
            now_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            section = (
                f"## {now_label}\n"
                f"- profile: `{self.profile_id}`\n"
                f"- source: `{self.daily_memory_path}`\n\n"
                f"{summary_text.strip()}\n\n"
            )
            with target.open("a", encoding="utf-8") as fp:
                fp.write(section)
            return True, str(target)
        except Exception as e:
            return False, str(e)

    def run(self):
        try:
            if not self.brain_model:
                self.completed.emit(False, "브레인 모델이 비어 있습니다.", {})
                return
            if not self.daily_memory_entry:
                self.completed.emit(False, "동기화할 Daily Memory 엔트리가 없습니다.", {})
                return
            manager = OpenClawGatewayManager()
            manager.settings = self.build_bridge_profile()
            source = str(manager.settings.get("modelSource") or "").strip().lower()
            if source == "ollama":
                exists_ok, exists_msg = verify_ollama_model_exists(manager.settings, self.brain_model)
                if not exists_ok:
                    self.completed.emit(False, exists_msg, {})
                    return
                ollama_ok, ollama_msg = send_direct_ollama_chat(
                    manager.settings,
                    self.brain_model,
                    (
                        "당신은 대화 메모 정리 요약기입니다. "
                        "도구 호출 없이, 핵심 사실/진행/다음 주입 추천을 한국어로 짧게 요약하세요."
                    ),
                    (
                        f"[profile]\\n{self.profile_id}\\n\\n"
                        f"[source_daily_file]\\n{self.daily_memory_path}\\n\\n"
                        "[latest_daily_memory_entry]\\n"
                        f"{self.daily_memory_entry}"
                    ),
                    max_tokens=220,
                )
                if not ollama_ok:
                    self.completed.emit(False, ollama_msg, {})
                    return
                save_ok, save_msg = self.append_summary_file(ollama_msg)
                if not save_ok:
                    self.completed.emit(False, f"요약 파일 저장 실패: {save_msg}", {})
                    return
                payload = {
                    "agentId": self.brain_agent_id,
                    "sessionKey": self.brain_session_key,
                    "model": self.brain_model,
                    "summaryPath": save_msg,
                }
                self.completed.emit(True, f"요약 저장 완료: {save_msg}", payload)
                return
            if not manager.check_gateway_running():
                self.completed.emit(False, "게이트웨이가 오프라인입니다.", {})
                return
            token = manager.get_gateway_token()
            if not token:
                self.completed.emit(False, "게이트웨이 토큰을 찾지 못했습니다.", {})
                return
            ok, message = self.send_sync_request(token)
            if not ok:
                if "does not support tools" in str(message or "").lower():
                    fallback_ok, fallback_msg = send_direct_ollama_chat(
                        manager.settings,
                        self.brain_model,
                        (
                            "당신은 대화 메모 정리 요약기입니다. "
                            "도구 호출 없이, 핵심 사실/진행/다음 주입 추천을 한국어로 짧게 요약하세요."
                        ),
                        (
                            f"[profile]\\n{self.profile_id}\\n\\n"
                            f"[source_daily_file]\\n{self.daily_memory_path}\\n\\n"
                            "[latest_daily_memory_entry]\\n"
                            f"{self.daily_memory_entry}"
                        ),
                        max_tokens=220,
                    )
                    if fallback_ok:
                        save_ok, save_msg = self.append_summary_file(fallback_msg)
                        if not save_ok:
                            self.completed.emit(False, f"요약 파일 저장 실패: {save_msg}", {})
                            return
                        payload = {
                            "agentId": self.brain_agent_id,
                            "sessionKey": self.brain_session_key,
                            "model": self.brain_model,
                            "summaryPath": save_msg,
                            "fallback": "direct-ollama",
                        }
                        self.completed.emit(True, f"요약 저장 완료: {save_msg}", payload)
                        return
                self.completed.emit(False, message, {})
                return
            save_ok, save_msg = self.append_summary_file(message)
            if not save_ok:
                self.completed.emit(False, f"요약 파일 저장 실패: {save_msg}", {})
                return
            payload = {
                "agentId": self.brain_agent_id,
                "sessionKey": self.brain_session_key,
                "model": self.brain_model,
                "summaryPath": save_msg,
            }
            self.completed.emit(True, f"요약 저장 완료: {save_msg}", payload)
        except Exception as e:
            self.completed.emit(False, str(e), {})


class GhostChatCompanionWindow(GhostChatWindow):
    brain_monitor_requested = Signal(str, str)
    brain_monitor_log = Signal(str, str)

    """
    Non-brain chat UI.
    - Keeps normal conversation on chat-only path
    - Keeps a separate brain bridge model/session warm and ready
    """

    def __init__(
        self,
        profile_id: Optional[str] = None,
        use_tray_icon: bool = True,
        console_managed: bool = False,
    ):
        self.brain_prepare_thread: Optional[BrainPrepareThread] = None
        self.brain_memory_sync_thread: Optional[BrainMemorySyncThread] = None
        self._bridge_loading = False
        self._bridge_prepared_once = False
        self._pending_daily_memory_entry = ""
        self._pending_daily_memory_signature = ""
        self._inflight_daily_memory_signature = ""
        self._last_synced_daily_memory_signature = ""
        self.brain_profile_id = ""
        self.brain_agent_id = ""
        self.brain_session_key = ""
        super().__init__(profile_id=profile_id, use_tray_icon=use_tray_icon, console_managed=console_managed)
        self.brain_memory_sync_timer = QTimer(self)
        self.brain_memory_sync_timer.setSingleShot(True)
        self.brain_memory_sync_timer.setInterval(1300)
        self.brain_memory_sync_timer.timeout.connect(self.flush_pending_brain_memory_sync)
        self.install_bridge_controls()
        self.load_brain_bridge_controls(update_status=False)
        self.force_chat_mode_runtime()
        if self.ready and hasattr(self, "brain_auto_prepare_check") and self.brain_auto_prepare_check.isChecked():
            QTimer.singleShot(120, lambda: self.prepare_brain_bridge(reason="startup"))

    def update_window_title(self):
        profile = self.profile_id
        self.setWindowTitle(f"💬 GhostRelay Chat Companion [{profile}]")
        if hasattr(self, "title_label"):
            self.title_label.setText(f"💬 Chat Companion [{profile}]")

    def refresh_settings_cache(self) -> dict:
        settings = super().refresh_settings_cache()
        self.force_chat_mode_runtime()
        return settings

    def reload_profile_runtime(self, profile_id: str, announce: bool = False):
        super().reload_profile_runtime(profile_id, announce=announce)
        self.force_chat_mode_runtime()
        self._pending_daily_memory_entry = ""
        self._pending_daily_memory_signature = ""
        self._inflight_daily_memory_signature = ""
        self._last_synced_daily_memory_signature = ""
        if hasattr(self, "brain_memory_sync_timer") and self.brain_memory_sync_timer.isActive():
            self.brain_memory_sync_timer.stop()
        if hasattr(self, "brain_profile_combo"):
            self.load_brain_bridge_controls(update_status=False)
            if self.ready and self.brain_auto_prepare_check.isChecked():
                QTimer.singleShot(120, lambda: self.prepare_brain_bridge(reason="profile-switch"))

    def on_startup_ready(self):
        super().on_startup_ready()
        if hasattr(self, "brain_auto_prepare_check") and self.brain_auto_prepare_check.isChecked():
            if not self._bridge_prepared_once:
                self._bridge_prepared_once = True
                QTimer.singleShot(120, lambda: self.prepare_brain_bridge(reason="startup-ready"))

    def force_chat_mode_runtime(self):
        if isinstance(self.profile_settings, dict):
            self.profile_settings["toolMode"] = "chat"

    def collect_brain_profile_candidates(self) -> list[str]:
        settings = load_settings()
        candidates = [normalize_agent_id(pid) for pid in list_profiles(settings)]
        if normalize_agent_id(self.profile_id) not in candidates:
            candidates.append(normalize_agent_id(self.profile_id))
        unique: list[str] = []
        seen = set()
        for item in candidates:
            raw = str(item or "").strip()
            if not raw:
                continue
            key = raw.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(raw)
        return unique

    def get_brain_bridge_config(self) -> dict:
        settings = load_settings()
        bridge = self.profile_settings.get("brainBridge", {})
        if not isinstance(bridge, dict):
            bridge = {}

        requested_profile = str(bridge.get("profileId") or "").strip()
        profile_id = normalize_agent_id(requested_profile or self.profile_id)
        available_profiles = set(self.collect_brain_profile_candidates())
        if profile_id not in available_profiles:
            profile_id = normalize_agent_id(self.profile_id)

        brain_profile_settings = get_profile(settings, profile_id)
        model = str(brain_profile_settings.get("model") or "google-vertex/gemini-2.0-flash").strip()
        agent_id = normalize_agent_id(str(brain_profile_settings.get("agentId") or profile_id))

        raw_session_key = str(bridge.get("sessionKey") or "brain").strip() or "brain"
        auto_prepare = bool(bridge.get("autoPrepare", True))
        return {
            "profileId": profile_id,
            "profileSettings": brain_profile_settings,
            "model": model,
            "agentId": agent_id,
            "rawSessionKey": raw_session_key,
            "sessionKey": build_agent_session_key(agent_id, raw_session_key),
            "autoPrepare": auto_prepare,
        }

    def install_bridge_controls(self):
        root = self.centralWidget().layout()
        if root is None:
            return

        bridge_widget = QWidget()
        bridge_widget.setStyleSheet("background: rgba(23, 27, 34, 0.88);")
        row = QHBoxLayout(bridge_widget)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(6)

        label = QLabel(self.tr("브레인 프로필", "Brain Profile"))
        label.setStyleSheet("color: rgba(255,255,255,0.76); font-size: 11px;")
        row.addWidget(label)

        self.brain_profile_combo = QComboBox()
        self.brain_profile_combo.setEditable(False)
        self.brain_profile_combo.setMinimumWidth(220)
        self.brain_profile_combo.setStyleSheet("""
            QComboBox, QLineEdit {
                background: rgba(14, 18, 24, 0.9);
                color: #e5e7eb;
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 6px;
                padding: 4px 6px;
                min-height: 24px;
                font-size: 11px;
            }
        """)
        row.addWidget(self.brain_profile_combo)

        self.brain_profile_model_label = QLabel("model: -")
        self.brain_profile_model_label.setStyleSheet("color: rgba(255,255,255,0.72); font-size: 11px;")
        row.addWidget(self.brain_profile_model_label, 1)

        self.brain_auto_prepare_check = QCheckBox(self.tr("자동 준비", "Auto Prepare"))
        self.brain_auto_prepare_check.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 11px;")
        row.addWidget(self.brain_auto_prepare_check)

        self.brain_prepare_button = QPushButton(self.tr("준비", "Prepare"))
        self.brain_prepare_button.setFixedHeight(24)
        self.brain_prepare_button.clicked.connect(lambda: self.prepare_brain_bridge(reason="manual"))
        row.addWidget(self.brain_prepare_button)

        self.brain_status_label = QLabel(self.tr("브레인 준비: 대기", "Brain: idle"))
        self.brain_status_label.setStyleSheet("color: #9ca3af; font-size: 11px;")
        row.addWidget(self.brain_status_label)

        root.insertWidget(1, bridge_widget)
        self.brain_bridge_widget = bridge_widget

        self.brain_profile_combo.currentIndexChanged.connect(self.on_brain_profile_changed)
        self.brain_auto_prepare_check.toggled.connect(self.on_brain_auto_prepare_toggled)

    def load_brain_bridge_controls(self, update_status: bool = True):
        if not hasattr(self, "brain_profile_combo"):
            return
        self._bridge_loading = True
        try:
            config = self.get_brain_bridge_config()
            candidates = self.collect_brain_profile_candidates()
            if config["profileId"] not in candidates:
                candidates.insert(0, config["profileId"])
            self.brain_profile_combo.blockSignals(True)
            self.brain_profile_combo.clear()
            for item in candidates:
                self.brain_profile_combo.addItem(item, item)
            idx = self.brain_profile_combo.findData(config["profileId"])
            if idx < 0 and self.brain_profile_combo.count() > 0:
                idx = 0
            if idx >= 0:
                self.brain_profile_combo.setCurrentIndex(idx)
            self.brain_profile_combo.blockSignals(False)
            self.brain_profile_model_label.setText(
                self.tr(
                    f"모델: {config['model']}",
                    f"Model: {config['model']}",
                )
            )
            self.brain_auto_prepare_check.blockSignals(True)
            self.brain_auto_prepare_check.setChecked(bool(config["autoPrepare"]))
            self.brain_auto_prepare_check.blockSignals(False)
            self.brain_profile_id = config["profileId"]
            self.brain_agent_id = config["agentId"]
            self.brain_session_key = config["sessionKey"]
            if update_status:
                self.brain_status_label.setText(
                    self.tr(
                        f"브레인 준비: {self.brain_profile_id} / 세션 {self.brain_session_key}",
                        f"Brain ready target: {self.brain_profile_id} / session {self.brain_session_key}",
                    )
                )
                self.brain_status_label.setStyleSheet("color: #9ca3af; font-size: 11px;")
        finally:
            self._bridge_loading = False

    def save_brain_bridge_config(self):
        if self._bridge_loading:
            return
        settings = load_settings()
        selected_profile = str(self.brain_profile_combo.currentData() or "").strip()
        if not selected_profile:
            selected_profile = str(self.brain_profile_combo.currentText() or "").strip()
        profile_id = normalize_agent_id(selected_profile or self.profile_id)
        available_profiles = set(list_profiles(settings))
        if profile_id not in available_profiles:
            profile_id = normalize_agent_id(self.profile_id)
        brain_profile_settings = get_profile(settings, profile_id)
        model = str(brain_profile_settings.get("model") or "google-vertex/gemini-2.0-flash").strip()
        auto_prepare = bool(self.brain_auto_prepare_check.isChecked())
        default_agent = normalize_agent_id(str(brain_profile_settings.get("agentId") or profile_id))
        current = self.get_brain_bridge_config()
        raw_session = current.get("rawSessionKey") or "brain"
        self.brain_agent_id = default_agent
        self.brain_session_key = build_agent_session_key(self.brain_agent_id, str(raw_session))
        self.brain_profile_id = profile_id
        self.merge_and_save_settings(
            {
                "brainBridge": {
                    "profileId": self.brain_profile_id,
                    "model": model,
                    "sessionKey": str(raw_session),
                    "autoPrepare": auto_prepare,
                }
            }
        )
        self.brain_profile_model_label.setText(
            self.tr(
                f"모델: {model}",
                f"Model: {model}",
            )
        )
        self.force_chat_mode_runtime()

    def on_brain_profile_changed(self):
        if self._bridge_loading:
            return
        self.save_brain_bridge_config()
        self.load_brain_bridge_controls(update_status=True)
        if self.brain_auto_prepare_check.isChecked():
            self.prepare_brain_bridge(reason="profile-change")

    def create_right_control_panel(self, targets: dict[str, Any]) -> QFrame:
        panel = QFrame()
        panel.setMinimumWidth(300)
        panel.setStyleSheet("""
            QFrame {
                background: rgba(18, 22, 30, 0.88);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 10px;
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
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title_label = QLabel(self.tr("메모리", "Memory"))
        title_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #dbe7ff;")
        layout.addWidget(title_label)

        desc = QLabel(
            self.tr(
                "Companion 모드는 메모리 파일 편집/동기화 중심으로 동작합니다.\n"
                "세션/스킬 설정은 브레인 채팅에서 관리하세요.",
                "Companion mode focuses on memory editing/sync.\n"
                "Manage session/skills in Brain Chat.",
            )
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: rgba(255,255,255,0.62); font-size: 10px;")
        layout.addWidget(desc)

        files_tabs = QTabWidget()
        files_tabs.addTab(
            self.create_editor_tab(
                "memory_daily",
                "Today",
                targets["memory_daily"],
                self.tr(
                    "오늘 대화 메모를 기록합니다. (workspace/memory/YYYY-MM-DD.md)",
                    "Write daily chat notes. (workspace/memory/YYYY-MM-DD.md)",
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
                    "브레인이 정리할 장기 기억 파일입니다. (MEMORY.md)",
                    "Long-term memory file maintained by brain sync. (MEMORY.md)",
                ),
            ),
            "MEMORY.md",
        )
        brain_summary_path = targets["memory_daily"].parent / "BRAIN_SUMMARY.md"
        files_tabs.addTab(
            self.create_editor_tab(
                "memory_brain_summary",
                "Brain Summary",
                brain_summary_path,
                self.tr(
                    "브레인 동기화 요약 로그 파일입니다. 자동으로 누적됩니다.",
                    "Brain sync summary log file. Entries are appended automatically.",
                ),
            ),
            self.tr("브레인 요약", "Brain Summary"),
        )
        layout.addWidget(files_tabs, 1)
        return panel

    def on_brain_auto_prepare_toggled(self, checked: bool):
        if self._bridge_loading:
            return
        _ = checked
        self.save_brain_bridge_config()
        if self.brain_auto_prepare_check.isChecked():
            self.prepare_brain_bridge(reason="auto-toggle")

    def set_brain_bridge_busy(self, busy: bool):
        enabled = not bool(busy)
        self.brain_profile_combo.setEnabled(enabled)
        self.brain_auto_prepare_check.setEnabled(enabled)
        self.brain_prepare_button.setEnabled(enabled)

    def build_brain_context_snapshot(self) -> str:
        lines: list[str] = []
        lines.append(f"profile={self.profile_id}")
        lines.append(f"brain_profile={getattr(self, 'brain_profile_id', self.profile_id)}")
        lines.append(f"chat_model={self.profile_settings.get('model')}")
        tail = self.chat_history[-8:] if isinstance(self.chat_history, list) else []
        if not tail:
            lines.append("recent_chat=(empty)")
        else:
            lines.append("recent_chat:")
            for item in tail:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "unknown")
                text = str(item.get("content") or "").replace("\n", " ").strip()
                if len(text) > 220:
                    text = text[:220] + "..."
                lines.append(f"- {role}: {text}")
        return "\n".join(lines)

    def prepare_brain_bridge(self, reason: str = "manual"):
        if self.brain_prepare_thread and self.brain_prepare_thread.isRunning():
            return
        if not self.gateway_manager.check_gateway_running():
            self.brain_status_label.setText(self.tr("브레인 준비 실패: 게이트웨이 오프라인", "Brain prepare failed: gateway offline"))
            self.brain_status_label.setStyleSheet("color: #ef4444; font-size: 11px;")
            return
        config = self.get_brain_bridge_config()
        model = str(config.get("model") or "").strip()
        if not model:
            self.brain_status_label.setText(self.tr("브레인 모델을 입력하세요.", "Enter brain model first."))
            self.brain_status_label.setStyleSheet("color: #ef4444; font-size: 11px;")
            return
        self.save_brain_bridge_config()
        config = self.get_brain_bridge_config()
        self.brain_profile_id = str(config.get("profileId") or self.profile_id)
        self.brain_agent_id = str(config.get("agentId") or self.brain_agent_id)
        self.brain_session_key = str(config.get("sessionKey") or self.brain_session_key)
        self.brain_monitor_requested.emit(str(self.brain_profile_id), "prepare")
        self.brain_monitor_log.emit(
            str(self.brain_profile_id),
            self.tr(
                f"[Companion] 브레인 준비 요청 ({reason}) / session={self.brain_session_key}",
                f"[Companion] Brain prepare requested ({reason}) / session={self.brain_session_key}",
            ),
        )
        self.set_brain_bridge_busy(True)
        self.brain_status_label.setText(
            self.tr(
                f"브레인 준비 중... ({self.brain_profile_id}, {reason})",
                f"Preparing brain... ({self.brain_profile_id}, {reason})",
            )
        )
        self.brain_status_label.setStyleSheet("color: #f59e0b; font-size: 11px;")
        context = self.build_brain_context_snapshot()
        self.brain_prepare_thread = BrainPrepareThread(
            profile_settings=config.get("profileSettings") if isinstance(config.get("profileSettings"), dict) else self.profile_settings,
            profile_id=self.brain_profile_id,
            brain_model=model,
            brain_agent_id=self.brain_agent_id,
            brain_session_key=self.brain_session_key,
            context_summary=context,
        )
        self.brain_prepare_thread.completed.connect(self.on_brain_prepare_completed)
        self.brain_prepare_thread.start()

    def on_brain_prepare_completed(self, ok: bool, message: str, payload_obj: object):
        self.set_brain_bridge_busy(False)
        self.brain_prepare_thread = None
        message_text = str(message or "").strip()
        if ok and is_error_like_brain_message(message_text):
            ok = False
        if ok:
            self.brain_status_label.setText(self.tr("브레인 준비 완료", "Brain ready"))
            self.brain_status_label.setStyleSheet("color: #22c55e; font-size: 11px;")
            preview = message_text.replace("\n", " ").strip()
            if len(preview) > 180:
                preview = preview[:180] + "..."
            self.brain_monitor_log.emit(
                str(getattr(self, "brain_profile_id", self.profile_id)),
                self.tr(
                    f"[Companion] 브레인 준비 완료: {preview}",
                    f"[Companion] Brain prepare done: {preview}",
                ),
            )
            self.append_debug(
                self.tr(
                    f"브레인 세션 준비됨: {self.brain_profile_id} / {self.brain_session_key} / {message_text}",
                    f"Brain session prepared: {self.brain_profile_id} / {self.brain_session_key} / {message_text}",
                )
            )
            return
        self.brain_monitor_log.emit(
            str(getattr(self, "brain_profile_id", self.profile_id)),
            self.tr(
                f"[Companion] 브레인 준비 실패: {message_text}",
                f"[Companion] Brain prepare failed: {message_text}",
            ),
        )
        self.brain_status_label.setText(self.tr("브레인 준비 실패", "Brain prepare failed"))
        self.brain_status_label.setStyleSheet("color: #ef4444; font-size: 11px;")
        self.append_system_line(f"⚠️ {self.tr('브레인 준비 실패', 'Brain prepare failed')}: {message_text}")

    def get_latest_daily_memory_entry(self) -> str:
        try:
            workspace_dir = self.get_active_workspace_dir()
            daily_path = workspace_dir / "memory" / f"{date.today().isoformat()}.md"
            if not daily_path.exists():
                return ""
            text = daily_path.read_text(encoding="utf-8", errors="ignore")
            if not text:
                return ""
            marker = "\n## "
            idx = text.rfind(marker)
            if idx >= 0:
                entry = "## " + text[idx + len(marker):]
            elif text.startswith("## "):
                entry = text
            else:
                return ""
            entry = entry.strip()
            if len(entry) > 2600:
                entry = entry[-2600:]
            return entry
        except Exception as e:
            self.append_debug(f"daily memory 최신 엔트리 읽기 실패: {e}")
            return ""

    def build_daily_memory_signature(self, entry: str) -> str:
        normalized = str(entry or "").strip()
        if not normalized:
            return ""
        return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()

    def queue_brain_memory_sync(self):
        config = self.get_brain_bridge_config()
        model = str(config.get("model") or "").strip()
        if not model:
            return
        entry = self.get_latest_daily_memory_entry()
        if not entry:
            return
        signature = self.build_daily_memory_signature(entry)
        if not signature:
            return
        if signature == self._last_synced_daily_memory_signature:
            return
        if signature == self._inflight_daily_memory_signature:
            return
        if signature == self._pending_daily_memory_signature:
            return
        self._pending_daily_memory_entry = entry
        self._pending_daily_memory_signature = signature
        if hasattr(self, "brain_memory_sync_timer"):
            self.brain_memory_sync_timer.start()

    def flush_pending_brain_memory_sync(self):
        if self.brain_prepare_thread and self.brain_prepare_thread.isRunning():
            self.brain_memory_sync_timer.start()
            return
        if self.brain_memory_sync_thread and self.brain_memory_sync_thread.isRunning():
            self.brain_memory_sync_timer.start()
            return
        if not self._pending_daily_memory_entry or not self._pending_daily_memory_signature:
            return
        if not self.gateway_manager.check_gateway_running():
            self.append_debug(self.tr("브레인 메모리 동기화 건너뜀: 게이트웨이 오프라인", "Skip brain memory sync: gateway offline"))
            self.brain_memory_sync_timer.start(3000)
            return
        config = self.get_brain_bridge_config()
        model = str(config.get("model") or "").strip()
        if not model:
            self.brain_memory_sync_timer.start(3000)
            return
        self.save_brain_bridge_config()
        config = self.get_brain_bridge_config()
        self.brain_profile_id = str(config.get("profileId") or self.profile_id)
        self.brain_agent_id = str(config.get("agentId") or self.brain_agent_id)
        self.brain_session_key = str(config.get("sessionKey") or self.brain_session_key)
        self.brain_monitor_requested.emit(str(self.brain_profile_id), "memory-sync")
        self.brain_monitor_log.emit(
            str(self.brain_profile_id),
            self.tr(
                f"[Companion] 메모리 동기화 요청 / session={self.brain_session_key}",
                f"[Companion] Memory sync requested / session={self.brain_session_key}",
            ),
        )
        entry = self._pending_daily_memory_entry
        signature = self._pending_daily_memory_signature
        self._pending_daily_memory_entry = ""
        self._pending_daily_memory_signature = ""
        self._inflight_daily_memory_signature = signature
        self.brain_memory_sync_thread = BrainMemorySyncThread(
            profile_settings=config.get("profileSettings") if isinstance(config.get("profileSettings"), dict) else self.profile_settings,
            profile_id=self.brain_profile_id,
            brain_model=model,
            brain_agent_id=self.brain_agent_id,
            brain_session_key=self.brain_session_key,
            daily_memory_entry=entry,
            daily_memory_path=str(self.get_active_workspace_dir() / "memory" / f"{date.today().isoformat()}.md"),
            summary_file_path=str(self.get_active_workspace_dir() / "memory" / "BRAIN_SUMMARY.md"),
        )
        self.brain_memory_sync_thread.completed.connect(self.on_brain_memory_sync_completed)
        self.brain_memory_sync_thread.start()

    def on_brain_memory_sync_completed(self, ok: bool, message: str, payload_obj: object):
        _ = payload_obj
        completed_signature = self._inflight_daily_memory_signature
        self._inflight_daily_memory_signature = ""
        self.brain_memory_sync_thread = None
        message_text = str(message or "").strip()
        if ok and is_error_like_brain_message(message_text):
            ok = False
        if ok:
            self._last_synced_daily_memory_signature = completed_signature
            preview = message_text.replace("\n", " ").strip()
            if len(preview) > 180:
                preview = preview[:180] + "..."
            self.brain_monitor_log.emit(
                str(getattr(self, "brain_profile_id", self.profile_id)),
                self.tr(
                    f"[Companion] 메모리 동기화 완료: {preview}",
                    f"[Companion] Memory sync done: {preview}",
                ),
            )
            self.append_debug(self.tr(f"브레인 메모리 동기화 완료: {preview}", f"Brain memory sync done: {preview}"))
        else:
            self.brain_monitor_log.emit(
                str(getattr(self, "brain_profile_id", self.profile_id)),
                self.tr(
                    f"[Companion] 메모리 동기화 실패: {message_text}",
                    f"[Companion] Memory sync failed: {message_text}",
                ),
            )
            self.append_debug(self.tr(f"브레인 메모리 동기화 실패: {message_text}", f"Brain memory sync failed: {message_text}"))
        if self._pending_daily_memory_signature and self._pending_daily_memory_signature != self._last_synced_daily_memory_signature:
            self.brain_memory_sync_timer.start()

    def on_agent_response(self, message: str):
        super().on_agent_response(message)
        self.queue_brain_memory_sync()

    def send_message(self):
        self.force_chat_mode_runtime()
        super().send_message()
