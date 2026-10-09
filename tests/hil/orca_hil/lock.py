"""Exclusive SDR lock: only one process can open the B206mini at a time."""
import fcntl
import os
from pathlib import Path


class SdrLockHeld(RuntimeError):
    pass


class SdrLock:
    """fcntl.flock-based lock held for the duration of a HIL session."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self._fd = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.pread(fd, 64, 0).decode(errors="replace").strip() or "unknown"
            os.close(fd)
            raise SdrLockHeld(f"SDR lock {self.path} is held by pid {holder}; "
                              "another HIL session or capture is running") from None
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
        self._fd = fd
        return self

    def release(self):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
