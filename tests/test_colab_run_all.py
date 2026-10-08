import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "colab_run_all.py"
spec = importlib.util.spec_from_file_location("colab_run_all", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make(tmp_path, tail=5):
    return mod.Pipeline(tmp_path / "logs" / "run.log", tail=tail)


def test_successful_stage_is_logged_and_counted(tmp_path, capsys):
    p = make(tmp_path)
    p.run("hello", "echo working; echo done")
    out = capsys.readouterr().out
    assert "ok hello" in out and p.rows[0][:2] == ("hello", "ok")
    assert "working" in (tmp_path / "logs" / "run.log").read_text()


def test_skip_condition_runs_nothing(tmp_path, capsys):
    p = make(tmp_path)
    p.run("cached", "echo should-not-run", skip_if=lambda: True)
    assert "already done" in capsys.readouterr().out
    assert "should-not-run" not in (tmp_path / "logs" / "run.log").read_text()
    assert p.rows[0][1] == "skipped"


def test_failure_stops_with_only_the_last_lines_and_a_summary(tmp_path, capsys):
    p = make(tmp_path, tail=3)
    p.run("first", "echo fine")
    with pytest.raises(SystemExit) as e:
        p.run("broken", "for i in 1 2 3 4 5 6 7 8; do echo line$i; done; echo 'Traceback: boom' >&2; exit 3")
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "broken FAILED" in out and "exit 3" in out
    assert "Traceback: boom" in out and "line1" not in out  # only the last 3 lines are shown
    assert "=== SUMMARY ===" in out and "STOPPED at: broken" in out
    assert "line1" in (tmp_path / "logs" / "run.log").read_text()  # the full log keeps everything


def test_progress_bars_with_carriage_returns_keep_the_last_state(tmp_path, capsys):
    p = make(tmp_path)
    with pytest.raises(SystemExit):
        p.run("bar", "printf '10%%\\r50%%\\r100%%\\n'; exit 1")
    assert "100%" in capsys.readouterr().out
