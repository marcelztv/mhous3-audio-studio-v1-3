from PySide6.QtCore import QObject, Signal

from . import settings as cfg


class Store(QObject):
    """The one shared settings dictionary every workspace edits."""
    changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.s = dict(cfg.DEFAULTS)

    def set(self, key, value):
        if self.s.get(key) != value:
            self.s[key] = value
            self.changed.emit(key)

    def load(self, d):
        self.s = dict(cfg.DEFAULTS)
        self.s.update(d)
        self.changed.emit("*")
