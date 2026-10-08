"""F1/F5 regression tests: bad CLI invocations must explain, not crash.

F1: `check-draft` without exactly one of --draft/--repo exits 2 with a
usage line (argparse), never a traceback.
F5: a missing model file names the label, echoes the path, and points at
`wirl recommend <hf-repo>` (where a file-less first-timer should go next).
"""
import pytest

from wirl import cli


def _stderr(capsys):
    return capsys.readouterr().err


# --- F1 ---------------------------------------------------------------------


def test_check_draft_without_a_source_is_a_usage_error(moe_model, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["check-draft", moe_model])
    assert e.value.code == 2
    err = _stderr(capsys)
    assert "usage:" in err
    assert "--draft" in err and "--repo" in err


def test_check_draft_with_both_sources_is_a_usage_error(moe_model, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["check-draft", moe_model,
                  "--draft", moe_model, "--repo", "someone/drafter-GGUF"])
    assert e.value.code == 2
    assert "usage:" in _stderr(capsys)


def test_check_draft_file_without_repo_is_a_usage_error(moe_model, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["check-draft", moe_model,
                  "--draft", moe_model, "--file", "drafter-Q4_K_M.gguf"])
    assert e.value.code == 2
    assert "--repo" in _stderr(capsys)


# --- F5 ---------------------------------------------------------------------


def test_missing_model_names_path_and_recommend(capsys):
    missing = "/nonexistent-model-12345.gguf"
    with pytest.raises(SystemExit) as e:
        cli.main(["inspect", missing, "--no-bandwidth"])
    # sys.exit(msg) carries the message as the exit payload; the top-level
    # handler prints it to stderr.
    assert e.value.code != 0
    assert missing in str(e.value.code)
    assert "recommend" in str(e.value.code)


def test_missing_draft_model_names_its_label(capsys):
    missing = "/nonexistent-draft-12345.gguf"
    with pytest.raises(SystemExit) as e:
        cli._load(missing, "draft model")
    assert e.value.code != 0
    assert "draft model" in str(e.value.code)
    assert missing in str(e.value.code)
    assert "recommend" in str(e.value.code)
