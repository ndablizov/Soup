"""#2044 — ``training.use_ring_attention: true`` is refused at config load.

``soup train`` validated the flag (CUDA, ``ring-flash-attn`` importable, two GPUs,
``max_length >= 8192``) and then no trainer on any task or backend read it: nothing
set up ring attention or sequence parallelism, so a run that passed the checks
trained with ordinary attention and said nothing. Like ``use_longlora`` (#1240) it is
refused at load until a trainer applies it.
"""

from __future__ import annotations

import ast
import pathlib
import typing

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig
from tests.conftest import strip_ansi

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "soup_cli"
TASKS = typing.get_args(SoupConfig.model_fields["task"].annotation)


def _yaml(value: str, *, task: str = "sft", backend: str = "transformers") -> str:
    return (
        f"base: some-org/some-model\ntask: {task}\nbackend: {backend}\n"
        "data:\n  train: d.jsonl\n  max_length: 16384\n"
        f"training:\n  use_ring_attention: {value}\n"
    )


def test_the_task_list_is_not_empty():
    # A loop over nothing passes.
    assert {"sft", "grpo", "dpo", "pretrain"} <= set(TASKS)


@pytest.mark.parametrize("task", TASKS)
def test_true_is_refused_on_every_task(task):
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml("true", task=task))
    assert "#2044" in str(excinfo.value)


@pytest.mark.parametrize("backend", ["transformers", "unsloth", "mlx"])
def test_true_is_refused_on_every_backend(backend):
    with pytest.raises(ValueError, match="#2044"):
        load_config_from_string(_yaml("true", backend=backend))


@pytest.mark.parametrize("spelling", ["true", "yes", "on", "1", "'t'", "True"])
def test_every_spelling_read_as_true_is_refused(spelling):
    with pytest.raises(ValueError, match="#2044"):
        load_config_from_string(_yaml(spelling))


@pytest.mark.parametrize("spelling", ["false", "no", "off", "0"])
def test_false_still_loads(spelling):
    assert load_config_from_string(_yaml(spelling)).training.use_ring_attention is False


def test_unset_still_loads():
    cfg = load_config_from_string("base: m\ntask: sft\ndata:\n  train: d.jsonl\n")
    assert cfg.training.use_ring_attention is False


def test_the_refusal_names_the_field_and_what_extends_context():
    with pytest.raises(ValidationError) as excinfo:
        SoupConfig(base="m", data={"train": "d.jsonl"}, training={"use_ring_attention": True})
    errors = excinfo.value.errors()
    assert [error["loc"] for error in errors] == [("training", "use_ring_attention")]
    message = errors[0]["msg"]
    assert "training.use_ring_attention: true is refused (#2044)" in message
    assert "ordinary attention" in message
    assert "rope_scaling_type" in message


def test_soup_train_stops_at_the_config(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "d.jsonl").write_text('{"instruction": "a", "output": "b"}\n', encoding="utf-8")
    (tmp_path / "soup.yaml").write_text(_yaml("true"), encoding="utf-8")
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run", "--yes"])
    assert result.exit_code != 0
    output = " ".join(strip_ansi(result.output).split())
    assert "#2044" in output, output


def _calls_to(name: str) -> list[str]:
    """Call sites of ``name`` anywhere under ``src/``."""
    sites = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if called == name:
                    sites.append(f"{path.relative_to(SRC)}:{node.lineno}")
    return sites


def test_nothing_applies_sequence_parallelism_yet():
    """Why the refusal stands; when a trainer starts applying it, remove the refusal.

    ``get_sequence_parallel_size`` is the one helper an implementation would need,
    and ``ring_flash_attn`` is imported only for the availability probe.
    """
    assert _calls_to("get_sequence_parallel_size") == []
    importers = sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if "import ring_flash_attn" in path.read_text(encoding="utf-8")
    )
    assert importers == ["utils/ring_attention.py"]
