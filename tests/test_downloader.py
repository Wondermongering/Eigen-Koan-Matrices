from tests import pytest
import os
import tempfile
from downloader import Downloader


def test_download_behavior():
    # Create a temporary file with more than 50 bytes
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"x" * 60)
        tmp_path = tmp.name

    try:
        downloader = Downloader()
        data = downloader.download(tmp_path)
        assert isinstance(data, (bytes, bytearray))
        assert len(data) >= 50
    finally:
        os.remove(tmp_path)
