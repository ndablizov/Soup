"""#1830 — ``soup draft measure`` / ``distill`` say which extra is missing.

On a core-only install the model-config read failed with
``could not read model config: No module named 'transformers'`` and nothing
else: ``_pair_vocab_sizes_or_fail`` (measure) and the distill call site caught
every exception alike and printed it. An ``ImportError`` is now reported with
the install hint the other commands use (``pip install "soup-cli[train]"``),
with the bracket intact: ``_fail`` escapes its message, so Rich cannot eat
``[train]`` and print a command that installs the package without the extra.
Exit code stays 1, and a non-import failure keeps its message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.commands import draft as draft_cmd
from soup_cli.commands.draft import app

from .conftest import strip_ansi


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _no_transformers(model_id: str, trc: bool = False) -> int:
    raise ModuleNotFoundError("No module named 'transformers'")


def _config_missing(model_id: str, trc: bool = False) -> int:
    raise OSError("model not found")


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    text = "".join(json.dumps(row) + "\n" for _ in range(20))
    (tmp_path / "p.jsonl").write_text(text, encoding="utf-8")
    return tmp_path


_MEASURE = ["measure", "--target", "org/target", "--draft", "org/tiny", "--prompts", "p.jsonl"]
_DISTILL = [
    "distill", "--target", "org/target", "--draft-base", "org/tiny",
    "--data", "p.jsonl", "-o", "draftout", "--plan-only",
]


@pytest.mark.parametrize("argv", [pytest.param(_MEASURE, id="measure"),
                                  pytest.param(_DISTILL, id="distill")])
def test_missing_transformers_prints_the_train_extra_hint(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    monkeypatch.setattr(draft_cmd, "_vocab_size_of", _no_transformers)
    result = CliRunner().invoke(app, argv)
    out = _plain(result.output)
    assert result.exit_code == 1, out
    assert "could not read model config" in out
    assert "No module named 'transformers'" in out
    # The bracket must survive Rich: `pip install "soup-cli"` would install the
    # package WITHOUT the extra the user is missing.
    assert 'pip install "soup-cli[train]"' in out


@pytest.mark.parametrize("argv", [pytest.param(_MEASURE, id="measure"),
                                  pytest.param(_DISTILL, id="distill")])
def test_control_a_non_import_failure_keeps_its_message_without_the_hint(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    monkeypatch.setattr(draft_cmd, "_vocab_size_of", _config_missing)
    result = CliRunner().invoke(app, argv)
    out = _plain(result.output)
    assert result.exit_code == 1, out
    assert "could not read model config: model not found" in out
    assert "soup-cli[train]" not in out
