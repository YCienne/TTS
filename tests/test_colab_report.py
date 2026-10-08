import json
import os
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "colab_report.py"
TOKEN = "github_pat_FAKESECRET_0123456789"


def make_remote(tmp_path, accept_pushes=True):
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    if not accept_pushes:
        hook = bare / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\necho 'remote: Permission to YCienne/TTS.git denied to YCienne.' >&2\nexit 1\n")
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC)
    return bare


class FakeGitHub:
    """Answers the two API calls the diagnosis makes; the write probe returns ``probe_status``."""

    def __init__(self, probe_status):
        outer = self

        class H(BaseHTTPRequestHandler):
            def _send(self, status, body, extra=None):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"full_name": "YCienne/TTS"},
                           {"github-authentication-token-expiration": "2026-10-15 12:00:00 UTC"})

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                msg = {403: "Resource not accessible by personal access token", 404: "Not Found",
                       401: "Bad credentials", 422: "Validation Failed"}[outer.probe_status]
                self._send(outer.probe_status, {"message": msg},
                           {"x-accepted-github-permissions": "contents=write"})

            def log_message(self, *a):
                pass

        self.probe_status = probe_status
        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()


def run_report(tmp_path, bare, api_url, files):
    env = dict(os.environ, REPORT_REMOTE=f"file://{bare}", REPORT_API=api_url, GITHUB_TOKEN=TOKEN)
    return subprocess.run([sys.executable, str(SCRIPT), "--name", "t", "--files", *map(str, files)],
                          capture_output=True, text=True, env=env)


@pytest.fixture
def log_file(tmp_path):
    f = tmp_path / "a.log"
    f.write_text("hello\n")
    return f


def test_successful_push_creates_the_branch_and_leaks_nothing(tmp_path, log_file):
    bare = make_remote(tmp_path)
    p = run_report(tmp_path, bare, "http://127.0.0.1:9", [log_file])
    assert p.returncode == 0 and "pushed 1 file(s) to branch colab-logs" in p.stdout
    branches = subprocess.run(["git", "-C", str(bare), "branch", "--list"], capture_output=True, text=True).stdout
    assert "colab-logs" in branches and "main" not in branches
    author = subprocess.run(["git", "-C", str(bare), "log", "colab-logs", "--format=%an"], capture_output=True, text=True).stdout
    assert author.strip() == "YCienne"
    assert TOKEN not in p.stdout + p.stderr


@pytest.mark.parametrize("status,expected", [
    (403, "may NOT write"),
    (404, "cannot see this repository"),
    (401, "rejected the token"),
    (422, "CAN write contents"),
])
def test_failed_push_is_diagnosed(tmp_path, log_file, status, expected):
    bare = make_remote(tmp_path, accept_pushes=False)
    gh = FakeGitHub(status)
    try:
        p = run_report(tmp_path, bare, gh.url, [log_file])
    finally:
        gh.close()
    out = p.stdout + p.stderr
    assert p.returncode == 1
    assert "git push failed" in out and "denied to YCienne" in out
    assert "fine-grained token" in out and "token expires 2026-10-15" in out
    assert expected in out
    assert TOKEN not in out
