from pathlib import Path


class DirectoryManager:
    _instance = None

    def __new__(cls, log_dir):
        if cls._instance is None:
            cls._instance = super(DirectoryManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_dir):
        if not hasattr(self, 'initialized'):
            base_log_dir = Path(log_dir).resolve()
            base_log_dir.mkdir(parents=True, exist_ok=True)

            # Determine a new versioned subdirectory under the base directory.
            # If base_log_dir/v0 does not exist, use it. Otherwise, try v1, v2, etc.
            version = 0
            while True:
                versioned_dir = base_log_dir / f'v{version}'
                if not versioned_dir.exists():
                    versioned_dir.mkdir(parents=True, exist_ok=True)
                    self.log_dir = versioned_dir
                    break
                version += 1

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