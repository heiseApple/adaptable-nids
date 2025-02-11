import time
from pathlib import Path


class DirectoryManager:
    _instance = None

    def __new__(cls, log_dir=None):
        if cls._instance is None:
            cls._instance = super(DirectoryManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_dir=None):
        if not hasattr(self, 'initialized'):
            base_log_dir = Path(log_dir).resolve()
            log_dir_ver = base_log_dir / f'{round(time.time())}'
            log_dir_ver.mkdir(parents=True, exist_ok=True)
            self.log_dir = log_dir_ver

            self.initialized = True

    def _ensure_directory(self, path):
        """
        Create the directory if it doesn't exist.
        """
        path.mkdir(parents=True, exist_ok=True)

    def mkdir(self, path):
        """
        Create a subdirectory under the root path if it does not exist.
        """
        full_path = self.log_dir / path
        self._ensure_directory(full_path)
        return full_path