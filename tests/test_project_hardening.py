from pathlib import Path


def test_makefile_contains_expected_targets() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    targets = [
        "install",
        "test",
        "lint",
        "format-check",
        "download-data",
        "inspect",
        "experiment",
        "all-experiments",
        "plots",
        "clean-outputs",
    ]

    for target in targets:
        assert f"{target}:" in makefile


def test_ci_workflow_runs_pytest_and_ruff() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert 'python-version: "3.11"' in workflow
    assert "ruff check" in workflow
    assert "ruff format --check" in workflow
    assert "pytest" in workflow
    assert "download_tu_dataset" not in workflow


def test_method_notes_are_committed_documentation() -> None:
    notes = Path("reports/method_notes.md")

    assert notes.is_file()
    contents = notes.read_text(encoding="utf-8")
    assert "Leakage-Safe C Tuning" in contents
    assert "Limitations" in contents
