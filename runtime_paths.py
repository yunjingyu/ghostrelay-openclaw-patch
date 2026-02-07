import os
from pathlib import Path
from typing import Iterable


GHOSTCHAT_DIR = Path(__file__).resolve().parent


def _unique_paths(candidates: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        try:
            resolved = item.resolve()
        except Exception:
            resolved = item
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def resolve_openclaw_dir() -> Path:
    cli_env = str(os.environ.get("OPENCLAW_CLI") or "").strip()
    if cli_env:
        cli_path = Path(cli_env)
        if cli_path.exists():
            if cli_path.is_file() and cli_path.name.lower() == "openclaw.mjs":
                return cli_path.parent.resolve()
            if cli_path.is_dir() and (cli_path / "openclaw.mjs").exists():
                return cli_path.resolve()

    root_env = str(os.environ.get("OPENCLAW_ROOT") or "").strip()
    if root_env:
        root_path = Path(root_env)
        if (root_path / "openclaw.mjs").exists():
            return root_path.resolve()
        if (root_path / "openclaw-main" / "openclaw.mjs").exists():
            return (root_path / "openclaw-main").resolve()

    candidate_roots = _unique_paths(
        [
            GHOSTCHAT_DIR,
            GHOSTCHAT_DIR.parent,
            Path.cwd(),
        ]
    )
    for root in candidate_roots:
        direct = root / "openclaw.mjs"
        nested = root / "openclaw-main" / "openclaw.mjs"
        if direct.exists():
            return root.resolve()
        if nested.exists():
            return (root / "openclaw-main").resolve()

    return (GHOSTCHAT_DIR / "openclaw-main").resolve()


def resolve_project_root() -> Path:
    openclaw_dir = resolve_openclaw_dir()
    if openclaw_dir.name.lower() == "openclaw-main":
        return openclaw_dir.parent.resolve()
    return GHOSTCHAT_DIR.resolve()


def resolve_gateway_script() -> Path:
    project_root = resolve_project_root()
    candidates = _unique_paths(
        [
            GHOSTCHAT_DIR / "start_gateway.bat",
            GHOSTCHAT_DIR / "tools" / "clawdbot" / "start_gateway.bat",
            project_root / "tools" / "clawdbot" / "start_gateway.bat",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

