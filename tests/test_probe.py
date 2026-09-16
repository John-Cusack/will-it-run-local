"""Hardware parsing against reported text, without querying real devices."""
from wirl import probe


def test_gpu_info_parses_uuid(monkeypatch):
    calls = []
    monkeypatch.setattr(probe.shutil, "which", lambda name: name)

    def run(cmd):
        calls.append(cmd)
        return "1, RTX 3090, 24576, 1234, 23342, 8.6, 595.84, 4, 16, 9751, 350, GPU-one\n"

    monkeypatch.setattr(probe, "_run", run)
    gpu = probe.gpu_info()[0]
    assert gpu["uuid"] == "GPU-one" and gpu["index"] == 1
    assert gpu["vram_used"] == 1234 << 20
    assert calls[0][1].endswith(",uuid")


def test_gpu_processes_parse_uuid(monkeypatch):
    monkeypatch.setattr(probe.shutil, "which", lambda name: name)
    calls = []

    def run(cmd):
        calls.append(cmd)
        return "42, llama-server, 1234, GPU-one\n"

    monkeypatch.setattr(probe, "_run", run)
    assert probe.gpu_processes() == [
        {"pid": "42", "name": "llama-server", "vram_mb": "1234", "gpu_uuid": "GPU-one"}]
    assert calls[0][1].endswith(",gpu_uuid")
