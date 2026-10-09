"""Optional local Windows OCR with a bounded, cancellable, hidden subprocess."""

import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

from .figure_evidence_contracts import OcrResult


class WindowsFigureOcr:
    def __init__(self, *, timeout_seconds=30):
        if not 0 < timeout_seconds <= 30:
            raise ValueError("OCR timeout must be at most 30 seconds")
        self.timeout_seconds = timeout_seconds

    def _executable(self):
        root = Path(os.environ.get("SYSTEMROOT", "C:/Windows"))
        executable = root / "System32/WindowsPowerShell/v1.0/powershell.exe"
        return executable if executable.is_file() else None

    @staticmethod
    def _result(status, code):
        return OcrResult(status=status, engine="Windows.Media.Ocr", error_code=code)

    def __call__(self, path):
        return self.recognize(path)

    def recognize(self, image_path, *, cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            return self._result("cancelled", "LOCAL_OCR_CANCELLED")
        if sys.platform != "win32" or self._executable() is None:
            return self._result("unavailable", "WINDOWS_OCR_UNAVAILABLE")
        path = Path(image_path).resolve()
        try:
            if (
                not path.is_file()
                or path.suffix.lower() != ".png"
                or path.stat().st_size > 10 * 1024 * 1024
            ):
                return self._result("failed", "LOCAL_OCR_IMAGE_REJECTED")
            content = path.read_bytes()
            if len(content) < 24 or content[:8] != b"\x89PNG\r\n\x1a\n":
                return self._result("failed", "LOCAL_OCR_IMAGE_REJECTED")
            width, height = struct.unpack(">II", content[16:24])
            if not width or not height or width * height > 4_000_000:
                return self._result("failed", "LOCAL_OCR_IMAGE_REJECTED")
            expected = hashlib.sha256(content).hexdigest()
            helper = Path(__file__).with_name("figure_evidence_windows_ocr.ps1")
            process = subprocess.Popen(
                [
                    str(self._executable()),
                    "-NoProfile",
                    "-NonInteractive",
                    "-File",
                    str(helper),
                    "-ImagePath",
                    str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            return self._result("unavailable", "WINDOWS_OCR_UNAVAILABLE")
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            cancelled = cancel_event is not None and cancel_event.is_set()
            remaining = deadline - time.monotonic()
            if cancelled or remaining <= 0:
                process.kill()
                process.communicate(timeout=5)
                return self._result(
                    "cancelled" if cancelled else "failed",
                    "LOCAL_OCR_CANCELLED" if cancelled else "LOCAL_OCR_TIMEOUT",
                )
            try:
                stdout, _stderr = process.communicate(timeout=min(0.2, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode != 0:
            return self._result("failed", "LOCAL_OCR_PROCESS_FAILED")
        try:
            if len(stdout.encode("utf-8")) > 1024 * 1024:
                raise ValueError("Oversized OCR output")
            response = json.loads(stdout)
            observed = response.pop("image_sha256", None)
            result = OcrResult.model_validate(response)
            if result.status == "ok" and (
                observed != expected
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError("OCR image identity differs")
            if any(
                word.x + word.width > width or word.y + word.height > height
                for word in result.words
            ):
                raise ValueError("OCR word outside source image")
            return result
        except (ValueError, OSError, TypeError, AttributeError):
            return self._result("failed", "LOCAL_OCR_INVALID_OUTPUT")
