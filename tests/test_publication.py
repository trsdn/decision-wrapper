import json
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

ROOT = Path(__file__).resolve().parents[1]


def result_paths():
    if not shutil.which("git") or not (ROOT / ".git").exists():
        return [ROOT / line[1:] for line in (ROOT / ".gitignore").read_text().splitlines()
                if line.startswith("!benchmark-results/") and not line.endswith("/")
                and (ROOT / line[1:]).is_file()]
    listed = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "benchmark-results"],
        cwd=ROOT, text=True,
    ).splitlines()
    return [ROOT / path for path in listed]


def test_only_reviewed_result_snapshots_are_publishable():
    if not shutil.which("git") or not (ROOT / ".git").exists():
        pytest.skip("git inclusion/exclusion checks require a repository checkout")
    paths = result_paths()
    assert len(paths) == 34
    assert all(path.suffix in {".json", ".jsonl", ".csv"} for path in paths)
    assert not any("runtimes" in path.relative_to(ROOT).parts for path in paths)
    for path in [
        "benchmark-results/runtimes/example/LICENSE",
        "benchmark-results/new-run/run.json",
        "benchmark-results/openai-decisions-20261009-105643/credentials.json",
        ".venv/example.py", ".env", ".env.local", "download.parquet", "model.safetensors",
    ]:
        assert subprocess.run(["git", "check-ignore", "-q", "--", path], cwd=ROOT).returncode == 0


def strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


@pytest.mark.parametrize("path", result_paths(), ids=lambda path: path.relative_to(ROOT).as_posix())
def test_published_artifacts_have_no_private_paths_or_credential_material(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        values = strings(json.loads(text))
    elif path.suffix == ".jsonl":
        values = (value for line in text.splitlines() if line.strip()
                  for value in strings(json.loads(line)))
    else:
        values = [text]
    secret = re.compile(
        r"(?<![A-Za-z0-9])(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"
        r"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
    )
    for value in values:
        if (secret.search(value) or "Bearer " in value or "Authorization" in value
                or "-----BEGIN PRIVATE KEY" in value):
            pytest.fail(f"credential-shaped material in {path.name}", pytrace=False)
        if PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute():
            pytest.fail(f"absolute private path in {path.name}", pytrace=False)


def test_third_party_dataset_notices_are_separate_from_project_license():
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")
    dataset_notice = (ROOT / "datasets" / "LICENSE").read_text()
    assert "rewardbench_sample.jsonl" in dataset_notice
    assert "https://opendatacommons.org/licenses/by/1-0/" in dataset_notice
    assert "not relicensed under MIT" in dataset_notice
