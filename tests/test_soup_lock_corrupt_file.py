"""#1830 — ``soup lock check`` / ``soup lock show`` name a corrupt lock file.

``read_lock`` called ``json.load`` bare, and ``json.JSONDecodeError`` is a
``ValueError``, so both commands printed the raw decoder text
(``Expecting value: line 1 column 1 (char 0)``), exit 3, without naming the file.
``read_lock`` now re-raises as ``ValueError`` with the path and "not valid JSON";
the decoder's own reason is kept after it. A file that is not UTF-8 is reported
the same way.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.exit_codes import EXIT_USAGE_ERROR
from soup_cli.utils.soup_lock import read_lock

from .conftest import strip_ansi

_CHECK_ARGS = [
    "--base-sha", "a" * 64,
    "--dataset-sha", "b" * 64,
    "--env-hash", "c" * 64,
    "--base-model", "org/m",
]


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture()
def corrupt_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "broken.lock").write_text("garbage: [", encoding="utf-8")
    return "broken.lock"


def test_read_lock_names_the_file_and_keeps_the_decoder_reason(corrupt_lock: str) -> None:
    with pytest.raises(ValueError) as exc:
        read_lock(corrupt_lock)
    msg = str(exc.value)
    assert "lock file 'broken.lock' is not valid JSON" in msg
    assert "Expecting value" in msg


def test_read_lock_reports_a_non_utf8_file_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bytes.lock").write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(ValueError, match=r"lock file 'bytes\.lock' is not valid JSON"):
        read_lock("bytes.lock")


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["lock", "show", "broken.lock"], id="show"),
        pytest.param(["lock", "check", "broken.lock", *_CHECK_ARGS], id="check"),
    ],
)
def test_cli_exits_3_and_names_the_corrupt_file(corrupt_lock: str, argv: list[str]) -> None:
    result = CliRunner().invoke(app, argv)
    out = _plain(result.output)
    assert result.exit_code == EXIT_USAGE_ERROR, out
    assert "broken.lock" in out
    assert "not valid JSON" in out


def test_control_a_missing_lock_file_keeps_its_own_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["lock", "show", "absent.lock"])
    out = _plain(result.output)
    assert result.exit_code == EXIT_USAGE_ERROR, out
    assert "lock file not found: 'absent.lock'" in out
    assert "not valid JSON" not in out
