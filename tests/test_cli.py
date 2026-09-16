"""Platform errors must precede hardware work, while argparse help still works."""
import subprocess
import sys

import pytest

from wirl import cli


def test_non_linux_command_is_a_clear_error(monkeypatch):
    # The negative control must never reach inventory or bandwidth measurement.
    monkeypatch.setattr(cli, "cmd_probe", lambda args: None)
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(SystemExit, match="will-it-run-local supports Linux only"):
        cli.main(["probe"])


@pytest.mark.parametrize("option", ["--help", "--version"])
def test_non_linux_help_and_version_work(monkeypatch, capsys, option):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(SystemExit) as error:
        cli.main([option])
    assert error.value.code == 0
    assert "wirl" in capsys.readouterr().out


def test_help_import_does_not_require_fcntl():
    code = """
import builtins
original = builtins.__import__
def imported(name, *args, **kwargs):
    if name == 'fcntl':
        raise ModuleNotFoundError('fcntl unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = imported
from wirl import cli
cli.main(['--help'])
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage: wirl" in result.stdout
