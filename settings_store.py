import json
import re
from pathlib import Path

SETTINGS_PATH = Path(__file__).parent / "settings.json"
REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_ID_RE = re.compile(r"[^a-z0-9_-]+")


def find_default_vertex_sa() -> str:
    ghostchat_dir = Path(__file__).resolve().parent
    candidates = [
        ghostchat_dir / "model" / "VertexGcp" / "black-alpha-486019-t6-bf161493c9d2.json",
        ghostchat_dir.parent / "model" / "VertexGcp" / "black-alpha-486019-t6-bf161493c9d2.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def normalize_profile_id(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    raw = PROFILE_ID_RE.sub("-", raw).strip("-")
    return raw or "main"


def normalize_agent_id(value: str | None, fallback: str = "main") -> str:
    return normalize_profile_id(value or fallback)


def default_workspace_for_profile(profile_id: str) -> str:
    normalized = normalize_profile_id(profile_id)
    if normalized == "main":
        return str((REPO_ROOT / "workspace").resolve())
    return str((REPO_ROOT / "workspace" / "profiles" / normalized).resolve())


def _normalize_workspace(profile_id: str, workspace_value: str | None) -> str:
    normalized_profile = normalize_profile_id(profile_id)
    raw = str(workspace_value or "").strip()
    if not raw:
        return default_workspace_for_profile(normalized_profile)
    try:
        resolved = Path(raw).resolve()
    except Exception:
        resolved = Path(raw)

    if normalized_profile == "main":
        return str(resolved)

    main_workspace = Path(default_workspace_for_profile("main")).resolve()
    if resolved == main_workspace:
        return default_workspace_for_profile(normalized_profile)
    return str(resolved)


def default_profile(profile_id: str = "main") -> dict:
    normalized = normalize_profile_id(profile_id)
    return {
        "model": "google-vertex/gemini-2.0-flash",
        "modelSource": "vertex",
        "toolMode": "auto",
        "sessionKey": "",
        "agentId": normalize_agent_id(normalized, "main"),
        "workspace": default_workspace_for_profile(normalized),
        "vertex": {
            "serviceAccountPath": find_default_vertex_sa(),
            "project": "black-alpha-486019-t6",
            "location": "us-central1",
        },
        "ollama": {
            "baseUrl": "http://127.0.0.1:11434/v1",
            "apiKey": "ollama-local",
            "modelName": "llama3.3",
            "contextWindow": 32768,
            "injectionLevel": "medium",
            "historyTurns": 6,
        },
        "brainBridge": {
            "profileId": normalized,
            "model": "google-vertex/gemini-2.0-flash",
            "agentId": normalize_agent_id(f"{normalized}-brain", f"{normalized}-brain"),
            "sessionKey": "brain",
            "autoPrepare": True,
        },
    }


def default_settings() -> dict:
    return {
        "activeProfile": "main",
        "alwaysOnTop": True,
        "ui": {
            "leftPanelVisible": False,
            "rightPanelVisible": True,
            "language": "ko",
        },
        "profiles": {
            "main": default_profile("main"),
        },
    }


def _merge_profile(base_profile: dict, payload: dict, profile_id: str) -> dict:
    merged = dict(base_profile)
    if not isinstance(payload, dict):
        return merged
    for key in ("model", "modelSource", "toolMode", "sessionKey", "workspace"):
        if key in payload:
            merged[key] = payload.get(key)
    if "agentId" in payload:
        merged["agentId"] = normalize_agent_id(str(payload.get("agentId") or ""), profile_id)
    if isinstance(payload.get("vertex"), dict):
        merged["vertex"].update(payload["vertex"])
    if isinstance(payload.get("ollama"), dict):
        merged["ollama"].update(payload["ollama"])
    if isinstance(payload.get("brainBridge"), dict):
        existing_bridge = merged.get("brainBridge", {})
        if not isinstance(existing_bridge, dict):
            existing_bridge = {}
        existing_bridge.update(payload["brainBridge"])
        merged["brainBridge"] = existing_bridge

    source = str(merged.get("modelSource") or "vertex").strip().lower()
    merged["modelSource"] = source if source in {"vertex", "ollama"} else "vertex"
    tool_mode = str(merged.get("toolMode") or "auto").strip().lower()
    merged["toolMode"] = tool_mode if tool_mode in {"auto", "chat", "agent"} else "auto"
    merged["model"] = str(merged.get("model") or "google-vertex/gemini-2.0-flash").strip()
    merged["sessionKey"] = str(merged.get("sessionKey") or "").strip()
    merged["workspace"] = _normalize_workspace(profile_id, merged.get("workspace"))
    merged["agentId"] = normalize_agent_id(str(merged.get("agentId") or ""), profile_id)
    if normalize_profile_id(profile_id) != "main" and merged["agentId"] == "main":
        merged["agentId"] = normalize_profile_id(profile_id)
    if not isinstance(merged.get("vertex"), dict):
        merged["vertex"] = default_profile(profile_id)["vertex"]
    if not isinstance(merged.get("ollama"), dict):
        merged["ollama"] = default_profile(profile_id)["ollama"]
    if not isinstance(merged.get("brainBridge"), dict):
        merged["brainBridge"] = default_profile(profile_id)["brainBridge"]
    brain_bridge = merged.get("brainBridge", {})
    if isinstance(brain_bridge, dict):
        bridge_profile = normalize_profile_id(
            str(brain_bridge.get("profileId") or normalize_profile_id(profile_id))
        )
        brain_bridge["profileId"] = bridge_profile
        brain_model = str(brain_bridge.get("model") or "").strip()
        if not brain_model:
            brain_model = "google-vertex/gemini-2.0-flash"
        brain_bridge["model"] = brain_model
        bridge_agent = normalize_agent_id(
            str(brain_bridge.get("agentId") or ""),
            f"{normalize_profile_id(profile_id)}-brain",
        )
        brain_bridge["agentId"] = bridge_agent
        session_key = str(brain_bridge.get("sessionKey") or "").strip()
        brain_bridge["sessionKey"] = session_key or "brain"
        brain_bridge["autoPrepare"] = bool(brain_bridge.get("autoPrepare", True))
    ollama = merged.get("ollama", {})
    if isinstance(ollama, dict):
        try:
            context_window = int(str(ollama.get("contextWindow") or "32768").strip())
        except Exception:
            context_window = 32768
        context_window = max(16000, min(262144, context_window))
        ollama["contextWindow"] = context_window
        injection_level = str(ollama.get("injectionLevel") or "medium").strip().lower()
        ollama["injectionLevel"] = injection_level if injection_level in {"weak", "medium", "strong"} else "medium"
        try:
            history_turns = int(str(ollama.get("historyTurns") or "6").strip())
        except Exception:
            history_turns = 6
        ollama["historyTurns"] = max(1, min(20, history_turns))
    return merged


def _inflate_active_profile_aliases(settings_root: dict) -> dict:
    active_id = normalize_profile_id(settings_root.get("activeProfile"))
    profiles = settings_root.get("profiles", {})
    active_profile = profiles.get(active_id)
    if not isinstance(active_profile, dict):
        active_id = "main" if "main" in profiles else next(iter(profiles.keys()), "main")
        active_profile = profiles.get(active_id, default_profile(active_id))
    settings_root["activeProfile"] = active_id
    settings_root["profileId"] = active_id
    settings_root["profile"] = active_profile
    for key in (
        "model",
        "modelSource",
        "toolMode",
        "sessionKey",
        "agentId",
        "workspace",
        "vertex",
        "ollama",
        "brainBridge",
    ):
        settings_root[key] = active_profile.get(key)
    return settings_root


def normalize_settings(raw: dict | None) -> dict:
    base = default_settings()
    if not isinstance(raw, dict):
        return _inflate_active_profile_aliases(base)

    next_settings = default_settings()
    if "alwaysOnTop" in raw:
        next_settings["alwaysOnTop"] = bool(raw.get("alwaysOnTop", True))
    if isinstance(raw.get("ui"), dict):
        next_settings["ui"].update(raw["ui"])

    profiles_payload = raw.get("profiles")
    if isinstance(profiles_payload, dict):
        profiles: dict[str, dict] = {}
        for profile_key, profile_payload in profiles_payload.items():
            normalized_id = normalize_profile_id(str(profile_key))
            merged_profile = _merge_profile(
                default_profile(normalized_id),
                profile_payload if isinstance(profile_payload, dict) else {},
                normalized_id,
            )
            profiles[normalized_id] = merged_profile
        if profiles:
            next_settings["profiles"] = profiles
    else:
        legacy_profile = _merge_profile(default_profile("main"), raw, "main")
        next_settings["profiles"] = {"main": legacy_profile}

    active_raw = str(raw.get("activeProfile") or "").strip()
    active_id = normalize_profile_id(active_raw) if active_raw else "main"
    if active_id not in next_settings["profiles"]:
        active_id = "main" if "main" in next_settings["profiles"] else next(iter(next_settings["profiles"]))
    next_settings["activeProfile"] = active_id
    return _inflate_active_profile_aliases(next_settings)


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return normalize_settings(None)
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return normalize_settings(data if isinstance(data, dict) else None)
    except Exception:
        return normalize_settings(None)


def to_storage_payload(settings: dict) -> dict:
    normalized = normalize_settings(settings)
    payload = {
        "activeProfile": normalized.get("activeProfile", "main"),
        "alwaysOnTop": bool(normalized.get("alwaysOnTop", True)),
        "ui": normalized.get("ui", {}),
        "profiles": normalized.get("profiles", {}),
    }
    return payload


def list_profiles(settings: dict | None = None) -> list[str]:
    source = settings if isinstance(settings, dict) else load_settings()
    profiles = source.get("profiles", {})
    if not isinstance(profiles, dict):
        return ["main"]
    keys = sorted(normalize_profile_id(key) for key in profiles.keys())
    return keys or ["main"]


def get_active_profile_id(settings: dict | None = None) -> str:
    source = settings if isinstance(settings, dict) else load_settings()
    return normalize_profile_id(source.get("activeProfile"))


def get_profile(settings: dict | None = None, profile_id: str | None = None) -> dict:
    source = settings if isinstance(settings, dict) else load_settings()
    target_id = normalize_profile_id(profile_id or source.get("activeProfile"))
    profiles = source.get("profiles", {})
    if not isinstance(profiles, dict):
        return default_profile(target_id)
    selected = profiles.get(target_id)
    if isinstance(selected, dict):
        return _merge_profile(default_profile(target_id), selected, target_id)
    return default_profile(target_id)


def save_settings(data: dict) -> None:
    if isinstance(data, dict) and "profiles" in data:
        normalized = normalize_settings(data)
        payload = to_storage_payload(normalized)
    else:
        current = load_settings()
        active_id = normalize_profile_id(current.get("activeProfile"))
        merged_profile = _merge_profile(
            default_profile(active_id),
            current.get("profile", {}),
            active_id,
        )
        if isinstance(data, dict):
            merged_profile = _merge_profile(merged_profile, data, active_id)
            if "alwaysOnTop" in data:
                current["alwaysOnTop"] = bool(data.get("alwaysOnTop"))
            if isinstance(data.get("ui"), dict):
                ui = current.get("ui", {}) if isinstance(current.get("ui"), dict) else {}
                ui.update(data["ui"])
                current["ui"] = ui
            next_profile = str(data.get("activeProfile") or "").strip()
            if next_profile:
                current["activeProfile"] = normalize_profile_id(next_profile)
                active_id = normalize_profile_id(current["activeProfile"])
        profiles = current.get("profiles", {})
        if not isinstance(profiles, dict):
            profiles = {}
        profiles[active_id] = merged_profile
        current["profiles"] = profiles
        payload = to_storage_payload(current)
    SETTINGS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
