import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
PYTHON = str((ROOT / "venv" / "Scripts" / "python.exe").resolve()) if os.name == "nt" else sys.executable
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def _create_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _is_ollama_ready(timeout_sec: float = 2.0) -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout_sec)
        response.raise_for_status()
        return True
    except Exception:
        return False


def _start_ollama() -> subprocess.Popen | None:
    if _is_ollama_ready():
        print("[run_all] Ollama is already ready.")
        return None

    print("[run_all] Starting ollama serve...")
    try:
        proc = subprocess.Popen(
            ["ollama", "serve"],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_create_flags(),
        )
    except Exception as exc:
        print(f"[run_all] Failed to start Ollama: {exc}")
        return None

    deadline = time.time() + 40
    while time.time() < deadline:
        if _is_ollama_ready():
            print("[run_all] Ollama is ready.")
            return proc
        time.sleep(1)

    print("[run_all] Ollama did not become ready in time.")
    return proc


def _start_named_process(name: str, script_name: str) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"{name}.log"
    err_path = LOG_DIR / f"{name}.err"
    out = open(out_path, "a", encoding="utf-8")
    err = open(err_path, "a", encoding="utf-8")
    print(f"[run_all] Starting {script_name}...")
    return subprocess.Popen(
        [PYTHON, script_name],
        cwd=str(ROOT),
        stdout=out,
        stderr=err,
        creationflags=_create_flags(),
    )


def _terminate(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=8)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main() -> int:
    print("[run_all] Root:", ROOT)
    print("[run_all] Python:", PYTHON)

    if not Path(PYTHON).exists():
        print("[run_all] venv python not found. Run: python -m venv venv && pip install -r requirements.txt")
        return 1

    ollama_proc = _start_ollama()
    web_proc = _start_named_process("web", "run_web.py")
    bot_proc = _start_named_process("bot", "run_bot.py")

    try:
        while True:
            if web_proc.poll() is not None:
                print("[run_all] web process exited, restarting...")
                time.sleep(2)
                web_proc = _start_named_process("web", "run_web.py")

            if bot_proc.poll() is not None:
                print("[run_all] bot process exited, restarting...")
                time.sleep(2)
                bot_proc = _start_named_process("bot", "run_bot.py")

            if not _is_ollama_ready():
                print("[run_all] Ollama unavailable, trying to start again...")
                _start_ollama()

            time.sleep(5)
    except KeyboardInterrupt:
        print("[run_all] Stopping...")
    finally:
        _terminate(web_proc)
        _terminate(bot_proc)
        _terminate(ollama_proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

