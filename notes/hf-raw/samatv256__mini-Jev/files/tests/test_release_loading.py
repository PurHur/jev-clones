"""Release-location tests; model construction is replaced to avoid GPU work."""

import json
import re
from pathlib import Path

import pytest
from huggingface_hub import constants
from huggingface_hub.errors import LocalEntryNotFoundError

import inference

FILES = (
    "config.json",
    "decision_head.safetensors",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
)


class ProbeMiniJev(inference.MiniJev):
    def __init__(self, release_dir, device="cuda"):
        self.release_dir = release_dir
        self.device = device
        self.config = json.loads((Path(release_dir) / "config.json").read_text())


@pytest.fixture
def release(tmp_path):
    for name in FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"marker": "resolved"}' if name.endswith('.json') else 'weights')
    return tmp_path


@pytest.fixture
def hub(monkeypatch, release):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(release)

    monkeypatch.setattr(inference, "snapshot_download", download, raising=False)
    return calls


def test_local_string_takes_precedence_over_repo_id(monkeypatch, tmp_path, release, hub):
    local = tmp_path / "example" / "model"
    local.mkdir(parents=True)
    (local / "config.json").write_text('{"marker": "local"}')
    monkeypatch.chdir(tmp_path)
    model = ProbeMiniJev.load("example/model", "cuda:1")
    assert model.config["marker"] == "local"
    assert model.release_dir == "example/model"
    assert model.device == "cuda:1"
    assert hub == []


def test_path_remains_local(release, hub):
    model = ProbeMiniJev.load(release)
    assert model.release_dir is release
    assert model.config["marker"] == "resolved"
    assert hub == []


@pytest.mark.parametrize("argument", [None, "samatv256/mini-Jev", "example/model"])
def test_hub_default_and_explicit_repo_use_exact_allowlist(argument, release, hub):
    model = ProbeMiniJev.load() if argument is None else ProbeMiniJev.load(argument)
    assert model.config["marker"] == "resolved"
    assert Path(model.release_dir) == release
    assert len(hub) == 1
    assert hub[0] == {
        "repo_id": argument or "samatv256/mini-Jev",
        "revision": None,
        "allow_patterns": list(FILES),
    }


@pytest.mark.parametrize("revision", ["step-010626", "a" * 40])
def test_revision_is_resolved_in_one_snapshot(revision, release, hub):
    model = ProbeMiniJev.load("example/model", revision=revision)
    assert model.config["marker"] == "resolved"
    assert len(hub) == 1
    assert hub[0]["revision"] == revision


@pytest.mark.parametrize("missing", FILES)
def test_incomplete_hub_snapshot_fails_before_constructor(missing, release, hub):
    (release / missing).unlink()
    with pytest.raises(FileNotFoundError, match="missing required release files:.*" + re.escape(missing)):
        ProbeMiniJev.load("example/model")


@pytest.mark.parametrize("argument", ["./missing/model", "../missing/model", "/missing/model", "missing", Path("missing/model")])
def test_explicit_missing_local_paths_do_not_download(argument, hub):
    with pytest.raises(FileNotFoundError):
        ProbeMiniJev.load(argument)
    assert hub == []


def test_local_revision_does_not_change_loading(release, hub):
    model = ProbeMiniJev.load(str(release), revision="step-010626")
    assert model.config["marker"] == "resolved"
    assert hub == []


def test_offline_cache_miss_is_propagated(monkeypatch):
    def download(**kwargs):
        raise LocalEntryNotFoundError("no cached release")

    monkeypatch.setattr(inference, "snapshot_download", download, raising=False)
    with pytest.raises(LocalEntryNotFoundError, match="no cached release"):
        ProbeMiniJev.load("example/model")


@pytest.mark.parametrize("revision", [None, "a" * 40])
def test_real_hub_offline_cache_hit(monkeypatch, tmp_path, release, revision):
    cache = tmp_path / "cache"
    storage = cache / "models--example--model"
    snapshot = storage / "snapshots" / ("a" * 40)
    entries = {}
    for name in FILES:
        data = (release / name).read_bytes()
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries[name] = {"size": len(data), "blob_id": "b" * 40}
    (storage / "refs").mkdir()
    (storage / "refs" / "main").write_text("a" * 40)
    (storage / "trees").mkdir()
    (storage / "trees" / (("a" * 40) + ".json")).write_text(
        json.dumps({"format_version": 1, "files": entries})
    )
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    model = ProbeMiniJev.load("example/model", revision=revision)
    assert Path(model.release_dir) == snapshot
    assert model.config["marker"] == "resolved"


def test_real_hub_offline_cache_miss(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(LocalEntryNotFoundError):
        ProbeMiniJev.load("example/uncached")
