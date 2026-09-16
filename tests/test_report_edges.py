"""Readable output, entry point and generated files."""
import runpy
import sys

import pytest

from wirl import doctor, emit, report

pytestmark = pytest.mark.usefixtures("isolated_runtime")


@pytest.mark.parametrize("tty,no_colour", [(True, False), (True, True), (False, False)])
def test_colour_respects_terminal_and_no_color(monkeypatch, tty, no_colour):
    monkeypatch.setattr(report.sys.stdout, "isatty", lambda: tty)
    monkeypatch.delenv("NO_COLOR", raising=False)
    if no_colour:
        monkeypatch.setenv("NO_COLOR", "")
    assert report.c("value", report.BOLD) == (report.BOLD+"value"+report.OFF if tty and not no_colour else "value")


def test_checks_hide_ok_and_wrap_fixes(capsys):
    checks = [doctor.Check("fine", "ok", "hidden", "unused"),
              doctor.Check("bad", "fail", "problem", "Repair this before measuring. "*5),
              doctor.Check("other", "custom", "detail")]
    report.print_checks(checks, show_ok=False)
    text = capsys.readouterr().out
    assert "hidden" not in text and "problem" in text and "Repair" in text
    assert len(text.splitlines()) > 3
    report.para("one two three", width=7, indent=">")
    assert capsys.readouterr().out == ">one two\n>three\n"
    assert report._wrap("", 10) == []


def test_long_word_does_not_add_an_empty_output_line():
    assert report._wrap("unbreakable next", 4) == ["unbreakable", "next"]


def test_module_entrypoint_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["wirl", "--help"])
    with pytest.raises(SystemExit) as exit:
        runpy.run_module("wirl", run_name="__main__")
    assert exit.value.code == 0 and "usage: wirl" in capsys.readouterr().out


def test_system_unit_and_nested_file(tmp_path):
    unit = emit.systemd_unit("/run.sh", "test", notes="one\n\ntwo", user_unit=False)
    assert "WantedBy=multi-user.target" in unit and "# two" in unit
    path = tmp_path / "nested" / "run.sh"
    assert emit.write(str(path), "text", 0o755) == str(path)
    assert path.read_text() == "text" and path.stat().st_mode & 0o111
