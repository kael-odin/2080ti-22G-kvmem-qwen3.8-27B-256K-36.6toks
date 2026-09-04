#!/usr/bin/env python3
"""Run one bounded, PID-scoped llama-server validation case on Windows."""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--messages", type=Path, help="OpenAI messages JSON array")
    parser.add_argument("--expect", action="append", default=[])
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--stage-mib", type=int, default=0)
    parser.add_argument("--cache-k", default="q4_0")
    parser.add_argument("--cache-v", default="q4_0")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--draft-n-max", type=int, default=2)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--output-root", type=Path, default=Path("validation-runs"))
    parser.add_argument("--label", default="case")
    return parser.parse_args()


def request_json(url: str, key: str, body: dict[str, Any] | None, timeout: int) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "kv-stream-sm75-validation/1.0",
        },
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"raw": raw}


def choose_port() -> int:
    for port in range(18080, 18090):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free validation port in 18080-18089")


def other_llama_processes() -> list[dict[str, Any]]:
    matches = []
    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
        try:
            info = proc.info
            name = (info.get("name") or "").lower()
            command = " ".join(info.get("cmdline") or []).lower()
            if name.startswith("llama-server") or "watchdog.bat" in command:
                matches.append(info)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return matches


def sample_gpu() -> tuple[int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        encoding="utf-8",
    ).strip()
    used, free = (int(item.strip()) for item in output.split(","))
    return used, free


def monitor(pid: int, path: Path, stop: threading.Event, summary: dict[str, int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unix_time", "rss_mib", "system_available_mib", "vram_used_mib", "vram_free_mib"])
        while not stop.is_set():
            try:
                rss = psutil.Process(pid).memory_info().rss // (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss = 0
            available = psutil.virtual_memory().available // (1024 * 1024)
            try:
                used, free = sample_gpu()
            except (OSError, subprocess.SubprocessError, ValueError):
                used, free = 0, 0
            writer.writerow([f"{time.time():.3f}", rss, available, used, free])
            handle.flush()
            summary["peak_rss_mib"] = max(summary.get("peak_rss_mib", 0), rss)
            summary["min_system_available_mib"] = min(summary.get("min_system_available_mib", available), available)
            summary["peak_vram_used_mib"] = max(summary.get("peak_vram_used_mib", 0), used)
            summary["min_vram_free_mib"] = min(summary.get("min_vram_free_mib", free), free)
            stop.wait(1)


def stop_tree(process: subprocess.Popen[Any]) -> None:
    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return
    members = parent.children(recursive=True) + [parent]
    for member in reversed(members):
        try:
            member.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(members, timeout=15)
    for member in alive:
        try:
            member.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=10)


def main() -> int:
    args = parse_args()
    server = args.server.resolve()
    model = args.model.resolve()
    mmproj = args.mmproj.resolve() if args.mmproj else None
    for path in (server, model, mmproj, args.messages):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if args.ubatch > args.batch or args.ctx <= 0 or args.stage_mib < 0:
        raise ValueError("require ctx > 0, stage-mib >= 0, and ubatch <= batch")

    existing = other_llama_processes()
    if existing:
        print(json.dumps({"error": "another llama-server or watchdog is active", "processes": existing}, default=str, indent=2))
        return 2

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"{stamp}-{args.label}"
    run_dir.mkdir()
    key = f"sk-local-{secrets.token_hex(16)}"
    port = choose_port()
    base = f"http://127.0.0.1:{port}"
    server_log = run_dir / "server.log"
    command = [
        str(server), "-m", str(model), "--alias", "kv-stream-test", "--api-key", key,
        "-c", str(args.ctx), "-ngl", "99", "-ctk", args.cache_k, "-ctv", args.cache_v,
        "--load-mode", "none", "-fa", "on", "--jinja", "--cache-ram", "0",
        "-b", str(args.batch), "-ub", str(args.ubatch), "-np", "1", "-t", "12",
        "-n", str(args.max_tokens), "--kv-stream-stage-mib", str(args.stage_mib),
        "--host", "127.0.0.1", "--port", str(port), "--timeout", str(args.request_timeout),
        "--log-file", str(server_log), "--verbose",
    ]
    if args.mtp:
        command += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.draft_n_max)]
    if mmproj:
        command += ["--mmproj", str(mmproj), "--image-min-tokens", "1024"]

    manifest = {"command": ["<redacted-local-key>" if item == key else item for item in command], "port": port}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    console = (run_dir / "console.log").open("wb")
    process: subprocess.Popen[Any] | None = None
    stop = threading.Event()
    resources: dict[str, int] = {}
    thread: threading.Thread | None = None
    result: dict[str, Any] = {"status": "failed", "run_dir": str(run_dir)}
    try:
        env = dict(**__import__("os").environ)
        env["PATH"] = f"{server.parent};{env.get('PATH', '')}"
        process = subprocess.Popen(
            command,
            cwd=server.parent,
            env=env,
            stdout=console,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
        )
        result["pid"] = process.pid
        thread = threading.Thread(target=monitor, args=(process.pid, run_dir / "resources.csv", stop, resources), daemon=True)
        thread.start()
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited during startup: {process.returncode}")
            try:
                status, payload = request_json(f"{base}/health", key, None, 3)
                if status == 200:
                    result["health"] = payload
                    break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError("health check timed out")

        if args.load_only:
            result["status"] = "passed"
        else:
            messages = json.loads(args.messages.read_text(encoding="utf-8")) if args.messages else [
                {"role": "user", "content": "Compute 19 + 23. Output only the number."}
            ]
            body = {
                "model": "kv-stream-test",
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "top_k": 1,
                "seed": 42,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            started = time.monotonic()
            status, response = request_json(f"{base}/v1/chat/completions", key, body, args.request_timeout)
            result["request_elapsed_s"] = round(time.monotonic() - started, 3)
            result["http_status"] = status
            result["timings"] = response.get("timings")
            result["usage"] = response.get("usage")
            if args.mtp:
                timings = result["timings"] or {}
                result["mtp"] = {
                    "draft_n": timings.get("draft_n"),
                    "draft_n_accepted": timings.get("draft_n_accepted"),
                }
            (run_dir / "response.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            result["expect"] = {needle: needle.lower() in content.lower() for needle in args.expect}
            checks_passed = all(result["expect"].values())
            mtp_passed = not args.mtp or bool((result.get("mtp") or {}).get("draft_n"))
            result["status"] = "passed" if status == 200 and checks_passed and mtp_passed else "failed"
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        stop.set()
        if thread:
            thread.join(timeout=5)
        if process:
            stop_tree(process)
        console.close()
        result["resources"] = resources
        log_text = server_log.read_text(encoding="utf-8", errors="replace") if server_log.exists() else ""
        result["adaptive_enabled"] = "experimental block KV streaming enabled" in log_text
        if args.stage_mib and not result["adaptive_enabled"]:
            result["status"] = "failed"
        (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
