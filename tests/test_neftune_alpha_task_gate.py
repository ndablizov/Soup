"""#1820 — ``training.neftune_alpha`` is applied by nine tasks; the other 14 refuse it.

The field had no task gate (``schema.py`` only bounds it, ``ge=0, le=50``), so
all 23 tasks loaded it while only seven wrappers forwarded it to
``TrainingArguments.neftune_noise_alpha``: ``sft.py`` (inherited by ``tts.py``),
``dpo.py``, ``ipo.py``, ``kto.py``, ``orpo.py``, ``simpo.py`` and ``bco.py``,
with ``preference`` delegating to one of the last six. Everywhere else the run
trained without NEFTune noise and nothing said so.

The task gate now refuses a set field outside ``NEFTUNE_APPLYING_TASKS``; these
tests pin the refusal per task, the applying set's continued acceptance, the
``null`` default's continued acceptance on every task, and that ``0.0`` counts
as set (the applying wrappers forward it on ``is not None``, so the gate uses
the same test).
"""

from __future__ import annotations

import re
import typing

import pytest
import yaml
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig

# The minimum each task needs to load at all (the same table as PR #1532's tests).
_TOP = {"tts": {"modality": "audio_out"}}
_DATA = {
    "dpo": {"format": "dpo"},
    "kto": {"format": "kto"},
    "pretrain": {"format": "plaintext"},
    "unlearn": {"forget_set": "./f.jsonl"},
}
_TRAINING = {
    "preference": {"preference_loss": "dpo"},
    "tts": {"tts_family": "orpheus"},
    "classifier": {"num_labels": 2},
    "reranker": {"num_labels": 2},
    "cross_encoder": {"num_labels": 2},
    "unlearn": {"unlearn_method": "npo"},
    "moe_lora_routing": {"mole_task_adapters": ["./a", "./b"]},
    "distill": {"teacher_model": "org/t"},
    "online_dpo": {"online_dpo_judge": "pairrm"},
}

TASKS = typing.get_args(SoupConfig.model_fields["task"].annotation)
# Listed literally, not read from NEFTUNE_APPLYING_TASKS: dropping a task from
# the constant must not also drop its control here (the mutation check the
# issue asks for: remove one task from the set and its test below fails).
APPLYING = ["sft", "tts", "dpo", "ipo", "kto", "orpo", "simpo", "bco", "preference"]
REFUSING = [t for t in TASKS if t not in APPLYING]


def _config_for(task: str, training_extra: dict) -> str:
    raw = {
        "base": "org/m",
        "task": task,
        **_TOP.get(task, {}),
        "data": {"train": "./x.jsonl", **_DATA.get(task, {})},
        "training": {**_TRAINING.get(task, {}), **training_extra},
    }
    return yaml.safe_dump(raw)


def test_the_two_task_lists_cover_every_task_exactly_once():
    # Guards the tables above against a task added to the schema later: it must
    # be sorted into one of the two lists by hand, not fall through untested.
    assert sorted(APPLYING + REFUSING) == sorted(TASKS)
    assert len(TASKS) == 23


@pytest.mark.parametrize("task", REFUSING)
def test_a_set_neftune_alpha_is_refused_outside_the_applying_tasks(task):
    with pytest.raises((ValidationError, ValueError)) as exc:
        load_config_from_string(_config_for(task, {"neftune_alpha": 5.0}))
    msg = str(exc.value)
    assert "training.neftune_alpha" in msg
    assert f"task={task!r}" in msg
    # The message names the tasks that do apply it, so the user can act on it.
    for applying_task in APPLYING:
        assert f"'{applying_task}'" in msg


@pytest.mark.parametrize("task", REFUSING)
def test_zero_counts_as_set_and_the_message_says_so(task):
    # The applying wrappers forward the field on ``is not None``, so ``0.0`` is
    # a value they would pass to TrainingArguments; the gate uses the same test
    # and tells the user to remove the key rather than zero it.
    with pytest.raises((ValidationError, ValueError)) as exc:
        load_config_from_string(_config_for(task, {"neftune_alpha": 0.0}))
    msg = str(exc.value)
    assert "training.neftune_alpha=0.0" in msg
    assert re.search(r"0\.0 (counts|included)", msg), msg
    assert "remove the key" in msg


@pytest.mark.parametrize("task", REFUSING)
def test_the_same_configs_load_without_the_field(task):
    cfg = load_config_from_string(_config_for(task, {}))
    assert cfg.task == task
    assert cfg.training.neftune_alpha is None


@pytest.mark.parametrize("task", APPLYING)
def test_the_applying_tasks_still_load_with_neftune_alpha_set(task):
    cfg = load_config_from_string(_config_for(task, {"neftune_alpha": 5.0}))
    assert cfg.training.neftune_alpha == 5.0
    assert cfg.task == task


@pytest.mark.parametrize("task", APPLYING)
def test_the_applying_tasks_still_load_with_neftune_alpha_zero(task):
    cfg = load_config_from_string(_config_for(task, {"neftune_alpha": 0.0}))
    assert cfg.training.neftune_alpha == 0.0


@pytest.mark.parametrize("task", TASKS)
def test_an_explicit_null_loads_on_every_task(task):
    # Dumped configs and recipe snapshots write ``neftune_alpha: null``; those
    # must keep loading everywhere (the recipe fixture carries exactly that).
    cfg = load_config_from_string(_config_for(task, {"neftune_alpha": None}))
    assert cfg.training.neftune_alpha is None


def test_the_bounds_still_apply_on_an_applying_task():
    # The gate sits beside the existing field bounds, not in place of them.
    with pytest.raises((ValidationError, ValueError)):
        load_config_from_string(_config_for("sft", {"neftune_alpha": 50.5}))
    with pytest.raises((ValidationError, ValueError)):
        load_config_from_string(_config_for("dpo", {"neftune_alpha": -1.0}))


def test_the_constant_matches_the_literal_list():
    # The schema constant and this file's literal list are two copies on
    # purpose; this pins that they have not drifted apart.
    from soup_cli.config.schema import NEFTUNE_APPLYING_TASKS

    assert NEFTUNE_APPLYING_TASKS == frozenset(APPLYING)
