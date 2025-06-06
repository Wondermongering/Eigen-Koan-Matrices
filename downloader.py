"""
Provides a simple mechanism for downloading/reading data from a local file path.
"""

class Downloader:
    """Simple data downloader that reads from a file path."""
    def download(self, path: str) -> bytes:
        """Return the bytes from the given file path."""
        with open(path, 'rb') as f:
            return f.read()

