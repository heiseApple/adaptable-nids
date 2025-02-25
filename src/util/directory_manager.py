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
            log_dir_ver = base_log_dir / f'{round(time.time())}' / 'src'
            log_dir_ver.mkdir(parents=True, exist_ok=True)
            self.log_dir = log_dir_ver

            # checkpoint_path is set when a network is saved on the disk (in base_network.py)
            self.checkpoint_path = None 
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
    
    def toggle_log_dir(self):
        """
        Updates the log directory path by toggling between 'src' and 'trg' directories.
        """
        if 'src' in self.log_dir.parts:
            new_dir = 'trg'
        elif 'trg' in self.log_dir.parts:
            new_dir = 'src'
        else:
            raise ValueError("The current log_dir does not contain 'src' or 'trg' and cannot be updated.")
        
        self.log_dir = self.log_dir.parent / new_dir
        self._ensure_directory(self.log_dir)
        