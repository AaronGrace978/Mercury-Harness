import json

from mercury.cli import main
from mercury.demo import frontier_auth_fix


def test_cli_demo_prints_pack(tmp_path, capsys):
    code = main(["--store", str(tmp_path / "s"), "demo"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Mercury Operating Pack" in captured.out
    assert "gpt-4o-mini" in captured.out


def test_cli_capture_status_and_pack(tmp_path, capsys):
    trace_path = tmp_path / "opus.json"
    trace_path.write_text(frontier_auth_fix().model_dump_json(), encoding="utf-8")
    store = str(tmp_path / "s")
    assert main(["--store", store, "init"]) == 0
    assert main(["--store", store, "capture", str(trace_path)]) == 0
    assert main(["--store", store, "status"]) == 0
    status = capsys.readouterr().out
    assert "frontier_traces" in status
    assert main(["--store", store, "pack", "--task", "fix login redirect", "--model", "gpt-4o-mini"]) == 0
    pack = capsys.readouterr().out
    assert "Operating Pack" in pack


def test_cli_pack_json(tmp_path, capsys):
    store = str(tmp_path / "s")
    main(["--store", store, "demo", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["stats"]["cards"] >= 1
    assert "markdown" in payload


def test_cli_grade_command(tmp_path, capsys):
    trace_path = tmp_path / "opus.json"
    trace_path.write_text(frontier_auth_fix().model_dump_json(), encoding="utf-8")
    store = str(tmp_path / "s")
    code = main(["--store", store, "grade", str(trace_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Behavior grade" in out
    assert "explored_first" in out
    code = main(["--store", store, "grade", str(trace_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["score"] >= 0.8
    assert payload["model"] == "claude-opus-4.1"
