import json

from secondbrain.cli import main


def test_cli_status_reports_runtime_state(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SECOND_BRAIN_DB_PATH", str(tmp_path / "secondbrain.sqlite"))

    exit_code = main(["status"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["node_count"] == 0
    assert payload["embedding_provider"] == "local"
