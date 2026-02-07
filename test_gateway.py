"""
게이트웨이 시작 테스트 스크립트
"""
import os
import sys
import subprocess
import time
import socket
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_paths import resolve_gateway_script, resolve_openclaw_dir, resolve_project_root


def decode_process_output(data: bytes | None) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", "cp949", "cp1252"):
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def run_process_capture(args: list[str], cwd: str, timeout: int = 20) -> tuple[int, str, str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=False,
        timeout=timeout,
    )
    return (
        result.returncode,
        decode_process_output(result.stdout),
        decode_process_output(result.stderr),
    )


def check_gateway_running():
    """게이트웨이 실행 중인지 확인"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 18789))
        sock.close()
        return result == 0
    except:
        return False

def test_gateway_start():
    """게이트웨이 시작 테스트"""
    repo_root = resolve_project_root()
    openclaw_dir = resolve_openclaw_dir()
    openclaw_mjs = openclaw_dir / "openclaw.mjs"
    gateway_bat = resolve_gateway_script()
    
    print(f"OpenClaw 경로: {openclaw_mjs}")
    print(f"존재 여부: {openclaw_mjs.exists()}")
    
    if not openclaw_mjs.exists():
        print("[ERROR] OpenClaw CLI를 찾을 수 없습니다")
        return False

    # Windows 배치 스크립트 우선 실행 (빌드/설정 포함)
    if sys.platform == "win32" and gateway_bat.exists():
        print(f"[INFO] gateway script 실행 시도: {gateway_bat}")
        try:
            subprocess.Popen(
                ["cmd", "/c", str(gateway_bat)],
                cwd=str(gateway_bat.parent),
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            # 포트 연결 대기
            for i in range(60):
                if check_gateway_running():
                    print(f"[OK] 게이트웨이 시작 완료! ({i * 0.5:.1f}초 소요)")
                    return True
                if i % 10 == 0 and i > 0:
                    print(f"  대기 중... ({i * 0.5:.1f}초)")
                time.sleep(0.5)
            print("[WARN] 게이트웨이 시작 타임아웃 (bat)")
        except Exception as e:
            print(f"[WARN] start_gateway.bat 실행 실패: {e}")

    # [FIX] Gateway start blocked 방지: gateway.mode local 강제
    try:
        run_process_capture(
            ["node", str(openclaw_mjs), "config", "set", "gateway.mode", "local"],
            cwd=str(openclaw_dir),
            timeout=20
        )
    except Exception as e:
        print(f"[WARN] gateway.mode 설정 실패 (무시 가능): {e}")

    # gateway.auth.token 설정 (token 모드 요구 시 필수)
    try:
        token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
        if not token:
            import uuid
            token = str(uuid.uuid4())
            os.environ["OPENCLAW_GATEWAY_TOKEN"] = token
        run_process_capture(
            ["node", str(openclaw_mjs), "config", "set", "gateway.auth.mode", "token"],
            cwd=str(openclaw_dir),
            timeout=20
        )
        run_process_capture(
            ["node", str(openclaw_mjs), "config", "set", "gateway.auth.token", token],
            cwd=str(openclaw_dir),
            timeout=20
        )
    except Exception as e:
        print(f"[WARN] gateway.auth.token 설정 실패 (무시 가능): {e}")
    
    # 이미 실행 중인지 확인
    if check_gateway_running():
        print("[OK] 게이트웨이가 이미 실행 중입니다")
        return True
    
    # 게이트웨이 시작
    print("\n게이트웨이 시작 중...")
    gateway_cmd = [
        "node", str(openclaw_mjs),
        "gateway", "run",
        "--dev",
        "--allow-unconfigured",
        "--force",
        "--port", "18789",
        "--bind", "loopback"
    ]
    
    print(f"명령어: {' '.join(gateway_cmd)}")
    print(f"작업 디렉토리: {openclaw_dir}")
    
    try:
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        process = subprocess.Popen(
            gateway_cmd,
            cwd=str(openclaw_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags
        )
        
        # 프로세스 시작 확인
        time.sleep(1)
        if process.poll() is not None:
            # 프로세스가 즉시 종료됨
            stdout, stderr = process.communicate()
            error_msg = stderr.decode('utf-8', errors='ignore') if stderr else stdout.decode('utf-8', errors='ignore')
            print(f"\n[ERROR] 게이트웨이 프로세스가 즉시 종료되었습니다")
            print(f"종료 코드: {process.returncode}")
            print(f"\n오류 메시지:")
            print(error_msg)
            return False
        
        print("[OK] 게이트웨이 프로세스가 시작되었습니다")
        print("포트 연결 대기 중... (최대 30초)")
        
        # 포트 연결 대기
        for i in range(60):
            if check_gateway_running():
                print(f"\n[OK] 게이트웨이 시작 완료! ({i * 0.5:.1f}초 소요)")
                print("프로세스 종료 중...")
                process.terminate()
                time.sleep(1)
                if process.poll() is None:
                    process.kill()
                return True
            if i % 10 == 0 and i > 0:
                print(f"  대기 중... ({i * 0.5:.1f}초)")
            time.sleep(0.5)
        
        print("\n[WARN] 게이트웨이 시작 타임아웃 (30초)")
        print("프로세스 종료 중...")
        process.terminate()
        time.sleep(1)
        if process.poll() is None:
            process.kill()
        return False
        
    except Exception as e:
        print(f"\n[ERROR] 게이트웨이 시작 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    print("=" * 50, file=sys.stderr)
    print("게이트웨이 시작 테스트", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    success = test_gateway_start()
    print("\n" + "=" * 50, file=sys.stderr)
    if success:
        print("[OK] 테스트 성공", file=sys.stderr)
    else:
        print("[ERROR] 테스트 실패", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    sys.exit(0 if success else 1)

