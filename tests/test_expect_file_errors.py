"""#1830 — ``soup expect`` names the problem when a data or suite file is missing.

``soup expect missing.jsonl suite.yaml`` and ``soup expect data.jsonl missing.yaml``
exited 3 (right) but printed only the bare path (``missing.jsonl``), because both
loaders raised ``FileNotFoundError(path)`` and the command prints ``str(exc)``.
The loaders now raise with a message that says what the path was for and what is
wrong with it; the not-a-regular-file branch uses the same wording.

Also pins the module docstring of ``commands/expect.py``: it said "Exit code 3 on
suite failure", while a suite failure exits 2 (``EXIT_GATE_FAILED``) and 3 is
``EXIT_USAGE_ERROR`` (``utils/exit_codes.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import expect as expect_module
from soup_cli.commands.expect import _load_jsonl_rows
from soup_cli.utils.exit_codes import EXIT_GATE_FAILED, EXIT_USAGE_ERROR
from soup_cli.utils.expectations import load_suite_yaml

from .conftest import strip_ansi

_SUITE = "expectations:\n  - name: expect_no_pii\n"
_ROW = '{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}\n'


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "suite.yaml").write_text(_SUITE, encoding="utf-8")
    (tmp_path / "data.jsonl").write_text(_ROW * 3, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# The loaders
# ---------------------------------------------------------------------------


def test_missing_data_file_names_the_problem_and_the_path(workdir: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"data file not found: 'missing\.jsonl'"):
        _load_jsonl_rows("missing.jsonl")


def test_data_path_that_is_a_directory_uses_the_same_wording(workdir: Path) -> None:
    (workdir / "adir").mkdir()
    with pytest.raises(FileNotFoundError, match=r"data path is not a regular file: 'adir'"):
        _load_jsonl_rows("adir")


def test_missing_suite_file_names_the_problem_and_the_path(workdir: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"suite file not found: 'missing\.yaml'"):
        load_suite_yaml("missing.yaml")


def test_suite_path_that_is_a_directory_uses_the_same_wording(workdir: Path) -> None:
    (workdir / "sdir").mkdir()
    with pytest.raises(FileNotFoundError, match=r"suite path is not a regular file: 'sdir'"):
        load_suite_yaml("sdir")


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def test_cli_missing_data_file_exits_3_with_the_message(workdir: Path) -> None:
    result = CliRunner().invoke(app, ["expect", "missing.jsonl", "suite.yaml"])
    out = _plain(result.output)
    assert result.exit_code == EXIT_USAGE_ERROR, out
    assert "data file not found: 'missing.jsonl'" in out


def test_cli_missing_suite_file_exits_3_with_the_message(workdir: Path) -> None:
    result = CliRunner().invoke(app, ["expect", "data.jsonl", "missing.yaml"])
    out = _plain(result.output)
    assert result.exit_code == EXIT_USAGE_ERROR, out
    assert "suite file not found: 'missing.yaml'" in out


def test_cli_control_a_present_pair_still_runs_the_suite(workdir: Path) -> None:
    result = CliRunner().invoke(app, ["expect", "data.jsonl", "suite.yaml"])
    out = _plain(result.output)
    assert result.exit_code == 0, out
    assert "not found" not in out


# ---------------------------------------------------------------------------
# The module docstring
# ---------------------------------------------------------------------------


def test_module_docstring_states_the_exit_codes_the_command_uses() -> None:
    doc = " ".join((expect_module.__doc__ or "").split())  # the docstring wraps lines
    assert "Exit code 3 on suite failure" not in doc
    assert f"Exit code {EXIT_GATE_FAILED} on suite failure" in doc
    assert f"{EXIT_USAGE_ERROR} on usage" in doc
