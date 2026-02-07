import fnmatch
import subprocess
import sys


BLOCKED_PATTERNS = [
    "settings.json",
    "logs/*",
    "__pycache__/*",
    ".venv/*",
    "workspace/*",
    "workspace/**",
    "profiles/*",
    "profiles/**",
    "memory/*",
    "memory/**",
    ".env",
    ".env.*",
    "*credentials*.json",
    "*service-account*.json",
    "*secret*.json",
    "*token*.json",
    "*.pem",
    "*.key",
    "*.gguf",
    "*.safetensors",
    "*.onnx",
]


def get_staged_files() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRTUXB"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    return [line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()]


def is_blocked(path: str) -> bool:
    normalized = path.lower()
    for pattern in BLOCKED_PATTERNS:
        if fnmatch.fnmatch(normalized, pattern.lower()):
            return True
    return False


def main() -> int:
    staged = get_staged_files()
    if not staged:
        return 0
    blocked = [path for path in staged if is_blocked(path)]
    if not blocked:
        return 0
    print("Blocked commit: private/sensitive files detected in staged changes.")
    for item in blocked:
        print(f" - {item}")
    print("Remove these files from commit and try again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
