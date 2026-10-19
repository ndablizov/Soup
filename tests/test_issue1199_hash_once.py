"""Issue #1199 fix 1 — a QuEST run reads each file of a local base once.

The pre-load pass (``SFTTrainerWrapper._setup_transformers``) hashed every selected
file of a local base, and the post-load check in ``_setup_quest`` (calibration and
continuation alike) resolved the identity again, reading and hashing every byte a
second time. Now the pre-load pass also keeps a stat snapshot of what it hashed, and
the post-load check compares that snapshot, reusing the pre-load digest.

These tests drive the real pre-load code (stopped at the tokenizer load, before any
model is built) and count ``_hash_local_file`` calls per path, the acceptance
criterion's own measure. The refusal a changed base gets is pinned for each way the
selection can change, and the one thing a stat comparison cannot see is pinned too,
so the trade-off is on record rather than implied.
"""

from __future__ import annotations

import collections
import copy
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import soup_cli.utils.quest as quest
from soup_cli.trainer.sft import SFTTrainerWrapper

CHANGED = "QuEST local base changed while the model was loading"


class _StopAtTokenizerError(Exception):
    """Raised by the patched tokenizer load: the pre-load pass has run by then."""


def _base(directory: Path, *, sharded: bool = False, weights: str = "model.safetensors") -> Path:
    directory.mkdir(parents=True)
    (directory / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (directory / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    if sharded:
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 2},
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors",
                    },
                }
            ),
            encoding="utf-8",
        )
        (directory / "model-00001-of-00002.safetensors").write_bytes(b"first")
        (directory / "model-00002-of-00002.safetensors").write_bytes(b"second")
    else:
        (directory / weights).write_bytes(b"safe-weights")
    return directory


@pytest.fixture
def hashed(monkeypatch):
    """Count ``_hash_local_file`` calls per path for the duration of a test."""
    real = quest._hash_local_file
    counts: collections.Counter[str] = collections.Counter()

    def counting(path: Path):
        counts[str(path)] += 1
        return real(path)

    monkeypatch.setattr(quest, "_hash_local_file", counting)
    return counts


def _preloaded(base: Path, monkeypatch, *, declaration=None) -> SFTTrainerWrapper:
    """A wrapper on which the real pre-load QuEST pass has run, and no model was built."""
    import transformers

    from soup_cli.trainer import stream_setup

    def stop(*_args, **_kwargs):
        raise _StopAtTokenizerError

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", stop)
    monkeypatch.setattr(quest, "validate_cuda_hardware", lambda: ("RTX 3090", (8, 6)))
    monkeypatch.setattr(stream_setup, "_distributed_launch", lambda: False)
    monkeypatch.setattr(quest, "calibration_rows_sha256", lambda rows: "ab" * 32)
    monkeypatch.setattr(
        quest,
        "calibrate_activation_scales",
        lambda model, rows: dict.fromkeys(quest.EXPECTED_MODULES, 3.0),
    )
    monkeypatch.setattr(
        quest,
        "install_mixed_quest",
        lambda model, *, activation_scales, base_model, calibration_sha256: (
            quest.build_metadata(
                activation_scales=activation_scales,
                base_model=base_model,
                calibration_sha256=calibration_sha256,
            )
        ),
    )
    monkeypatch.setattr(quest, "restore_mixed_quest", lambda model, metadata: model)

    wrapper = object.__new__(SFTTrainerWrapper)
    wrapper.config = SimpleNamespace(base=str(base))
    wrapper._trust_remote_code = False
    wrapper.deepspeed_config = None
    wrapper.fsdp_config = None
    tcfg = SimpleNamespace(quantization_aware="quest", use_liger=False, use_cut_ce=False)
    with pytest.raises(_StopAtTokenizerError):
        wrapper._setup_transformers(SimpleNamespace(base=str(base)), tcfg)
    wrapper.model = SimpleNamespace(config=SimpleNamespace(soup_quest=declaration))
    return wrapper


ROWS = [{"input_ids": [1]}] * quest.CALIBRATION_EXAMPLES


@pytest.mark.parametrize("sharded", [False, True], ids=["single-file", "sharded"])
def test_a_calibration_run_hashes_each_selected_file_once(tmp_path, monkeypatch, hashed, sharded):
    base = _base(tmp_path / "base", sharded=sharded)
    wrapper = _preloaded(base, monkeypatch)
    wrapper._setup_quest(ROWS)

    selected = {str(path) for _, path in quest._selected_local_model_files(base.resolve())}
    assert set(hashed) == selected
    assert sorted(hashed.values()) == [1] * len(selected), sorted(hashed.items())
    # The identity written into the artifact is the one the single pass computed.
    assert wrapper._quest_metadata["base_model"] == wrapper._quest_base_identity_before
    assert wrapper._quest_base_identity_before == quest.resolve_base_model_identity(str(base))


def test_a_continuation_run_hashes_each_selected_file_once(tmp_path, monkeypatch, hashed):
    base = _base(tmp_path / "artifact")
    declaration = quest.build_metadata(
        activation_scales=dict.fromkeys(quest.EXPECTED_MODULES, 3.0),
        base_model="local-sha256:" + "b" * 64,
        calibration_sha256="c" * 64,
    )
    quest.write_metadata(base, declaration)
    wrapper = _preloaded(base, monkeypatch, declaration=copy.deepcopy(declaration))
    wrapper._setup_quest([])

    assert wrapper._quest_metadata == declaration
    assert hashed and set(hashed.values()) == {1}, sorted(hashed.items())


def _rewrite_weights(base: Path) -> None:
    (base / "model.safetensors").write_bytes(b"changed-after-model-load")


def _touch_same_size(base: Path) -> None:
    # Same bytes, same size: only the modification time moves.
    path = base / "model.safetensors"
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))


def _add_safetensors_beside_bin(base: Path) -> None:
    (base / "model.safetensors").write_bytes(b"would now be selected")


def _add_tokenizer_file(base: Path) -> None:
    (base / "special_tokens_map.json").write_text("{}", encoding="utf-8")


def _remove_tokenizer_file(base: Path) -> None:
    (base / "tokenizer.json").unlink()


def _add_chat_template(base: Path) -> None:
    (base / "additional_chat_templates").mkdir()
    (base / "additional_chat_templates" / "tool_use.jinja").write_text("{{ m }}", encoding="utf-8")


def _replace_directory(base: Path) -> None:
    # Same path, same bytes, but another directory: every file is a new inode.
    moved = base.with_name("moved-away")
    base.rename(moved)
    shutil.copytree(moved, base)


CHANGES = {
    "weights rewritten": ("model.safetensors", _rewrite_weights),
    "weights touched, same size": ("model.safetensors", _touch_same_size),
    "safetensors added beside a bin base": ("pytorch_model.bin", _add_safetensors_beside_bin),
    "tokenizer file added": ("model.safetensors", _add_tokenizer_file),
    "tokenizer file removed": ("model.safetensors", _remove_tokenizer_file),
    "chat template added": ("model.safetensors", _add_chat_template),
    "base directory replaced": ("model.safetensors", _replace_directory),
}


@pytest.mark.parametrize("change", list(CHANGES), ids=list(CHANGES))
def test_a_base_changed_during_the_load_is_refused(tmp_path, monkeypatch, change):
    weights, mutate = CHANGES[change]
    base = _base(tmp_path / "base", weights=weights)
    wrapper = _preloaded(base, monkeypatch)
    mutate(base)
    with pytest.raises(ValueError, match=CHANGED):
        wrapper._setup_quest(ROWS)


def _swap_shard(base: Path) -> None:
    shard = base / "model-00002-of-00002.safetensors"
    shard.write_bytes(b"SECOND")  # same size, other bytes
    os.utime(shard, ns=(1, 1))


def _delete_shard(base: Path) -> None:
    (base / "model-00002-of-00002.safetensors").unlink()


@pytest.mark.parametrize("mutate", [_swap_shard, _delete_shard], ids=["swapped", "deleted"])
def test_a_changed_shard_is_refused(tmp_path, monkeypatch, mutate):
    base = _base(tmp_path / "base", sharded=True)
    wrapper = _preloaded(base, monkeypatch)
    mutate(base)
    with pytest.raises(ValueError, match=CHANGED):
        wrapper._setup_quest(ROWS)


def test_a_file_changed_between_its_snapshot_stat_and_its_hash_is_refused(tmp_path, monkeypatch):
    # The snapshot must describe the bytes that were hashed, not a later state.
    base = _base(tmp_path / "base")
    real = quest._hash_local_file

    def change_first(path: Path):
        if path.name == "model.safetensors":
            path.write_bytes(b"rewritten just before the hash")
        return real(path)

    monkeypatch.setattr(quest, "_hash_local_file", change_first)
    with pytest.raises(ValueError, match="changed during fingerprinting"):
        quest.resolve_base_model_identity_and_snapshot(str(base))


def test_a_continuation_base_changed_during_the_load_is_refused(tmp_path, monkeypatch):
    base = _base(tmp_path / "artifact")
    declaration = quest.build_metadata(
        activation_scales=dict.fromkeys(quest.EXPECTED_MODULES, 3.0),
        base_model="local-sha256:" + "b" * 64,
        calibration_sha256="c" * 64,
    )
    quest.write_metadata(base, declaration)
    wrapper = _preloaded(base, monkeypatch, declaration=copy.deepcopy(declaration))
    _rewrite_weights(base)
    with pytest.raises(ValueError, match=CHANGED):
        wrapper._setup_quest([])


def test_control_a_file_the_selector_ignores_does_not_refuse(tmp_path, monkeypatch, hashed):
    base = _base(tmp_path / "base")
    wrapper = _preloaded(base, monkeypatch)
    (base / "README.md").write_text("not loaded", encoding="utf-8")
    (base / "pytorch_model.bin").write_bytes(b"loses to model.safetensors")
    wrapper._setup_quest(ROWS)
    assert set(hashed.values()) == {1}


def test_control_a_hub_base_has_no_snapshot_and_keeps_its_id(tmp_path, monkeypatch, hashed):
    monkeypatch.chdir(tmp_path)
    wrapper = _preloaded(Path("ahxt/LiteLlama-460M-1T"), monkeypatch)
    assert wrapper._quest_base_snapshot is None
    wrapper._setup_quest(ROWS)
    assert wrapper._quest_metadata["base_model"] == "ahxt/LiteLlama-460M-1T"
    assert not hashed


def test_the_snapshot_stats_what_the_hash_read_and_reads_no_content(tmp_path, monkeypatch):
    base = _base(tmp_path / "base", sharded=True)
    identity, snapshot = quest.resolve_base_model_identity_and_snapshot(str(base))
    assert identity == quest.resolve_base_model_identity(str(base))
    assert [name for name, _ in snapshot.files] == [
        name for name, _ in quest._selected_local_model_files(base.resolve())
    ]
    # The post-load check opens nothing: any read would go through Path.open / read_*.
    for method in ("open", "read_text", "read_bytes"):
        monkeypatch.setattr(Path, method, lambda *a, **k: pytest.fail("read file content"))
    assert quest.local_base_unchanged(str(base), snapshot)


def test_a_path_that_is_no_longer_a_directory_is_not_unchanged(tmp_path):
    base = _base(tmp_path / "base")
    _, snapshot = quest.resolve_base_model_identity_and_snapshot(str(base))
    shutil.rmtree(base)
    assert not quest.local_base_unchanged(str(base), snapshot)
    base.write_text("now a file", encoding="utf-8")
    assert not quest.local_base_unchanged(str(base), snapshot)


def test_the_stat_check_cannot_see_a_same_size_rewrite_that_restores_the_mtime(tmp_path):
    """The trade-off #1199 fix 1 accepts, pinned so it is stated rather than implied.

    A content re-hash would refuse this; the stat comparison does not. It is enough
    for the accidental "the base changed while it was loading" case the check exists
    for, and the identity recorded in the artifact is still the content digest.
    """
    base = _base(tmp_path / "base")
    _, snapshot = quest.resolve_base_model_identity_and_snapshot(str(base))
    path = base / "model.safetensors"
    info = path.stat()
    with path.open("r+b") as stream:
        stream.write(b"SAFE")
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert quest.local_base_unchanged(str(base), snapshot)
