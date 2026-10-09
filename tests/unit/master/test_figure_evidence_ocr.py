"""Adapter control tests with local subprocess doubles, not OCR quality tests."""

import json
import subprocess
import threading

import pymupdf
import pytest

from materials_screening.master.figure_evidence_ocr import WindowsFigureOcr


@pytest.fixture
def image(tmp_path):
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=100, height=100)
        page.insert_text((10, 30), "unit")
        path = tmp_path / "unit.png"
        page.get_pixmap().save(path)
    return path


def test_non_windows_is_optional(monkeypatch, image):
    monkeypatch.setattr(
        "materials_screening.master.figure_evidence_ocr.sys.platform", "linux"
    )
    assert WindowsFigureOcr().recognize(image).status == "unavailable"


def test_cancelled_adapter_does_not_spawn(image):
    event = threading.Event()
    event.set()
    assert WindowsFigureOcr().recognize(image, cancel_event=event).status == "cancelled"


def test_timeout_is_bounded_and_process_cleaned(monkeypatch, image):
    module = "materials_screening.master.figure_evidence_ocr."
    monkeypatch.setattr(module + "sys.platform", "win32")
    monkeypatch.setattr(module + "WindowsFigureOcr._executable", lambda self: image)

    class Process:
        killed = False
        returncode = 0

        def communicate(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("owned-test-process", timeout)
            return "", ""

        def kill(self):
            self.killed = True

    process = Process()
    monkeypatch.setattr(module + "subprocess.Popen", lambda *a, **k: process)
    result = WindowsFigureOcr(timeout_seconds=0.01).recognize(image)
    assert result.status == "failed" and result.error_code == "LOCAL_OCR_TIMEOUT"
    assert process.killed


def test_invalid_output_never_exposes_process_details(monkeypatch, image):
    module = "materials_screening.master.figure_evidence_ocr."
    monkeypatch.setattr(module + "sys.platform", "win32")
    monkeypatch.setattr(module + "WindowsFigureOcr._executable", lambda self: image)

    def popen(args, **kw):
        assert kw["shell"] is False
        assert "-NoProfile" in args and "-NonInteractive" in args
        return type(
            "Process",
            (),
            {
                "returncode": 0,
                "communicate": lambda self, **k: ("private garbage", "private errors"),
            },
        )()

    monkeypatch.setattr(module + "subprocess.Popen", popen)
    result = WindowsFigureOcr().recognize(image)
    assert result.error_code == "LOCAL_OCR_INVALID_OUTPUT"
    assert "private" not in result.model_dump_json()


def test_valid_result_requires_input_digest(monkeypatch, image):
    import hashlib

    module = "materials_screening.master.figure_evidence_ocr."
    monkeypatch.setattr(module + "sys.platform", "win32")
    monkeypatch.setattr(module + "WindowsFigureOcr._executable", lambda self: image)
    response = dict(
        status="ok",
        engine="Windows.Media.Ocr",
        language="unit",
        raw_text="1 80",
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        words=[],
    )

    def popen(*args, **kwargs):
        return type(
            "Process",
            (),
            {
                "returncode": 0,
                "communicate": lambda self, **k: (json.dumps(response), ""),
            },
        )()

    monkeypatch.setattr(module + "subprocess.Popen", popen)
    assert WindowsFigureOcr().recognize(image).raw_text == "1 80"
    response["image_sha256"] = "a" * 64
    assert WindowsFigureOcr().recognize(image).error_code == "LOCAL_OCR_INVALID_OUTPUT"
