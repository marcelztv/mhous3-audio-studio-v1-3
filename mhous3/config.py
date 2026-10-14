"""One lightweight JSON config for theme, layouts, shortcuts and first-run flags."""
import json

from .settings import user_dir


class Config:
    def __init__(self):
        self.path = user_dir() / "config.json"
        try:
            self.d = json.loads(self.path.read_text("utf-8"))
        except Exception:
            self.d = {}

    def get(self, key, default=None):
        return self.d.get(key, default)

    def set(self, key, value, save=True):
        self.d[key] = value
        if save:
            self.save()

    def save(self):
        try:
            self.path.write_text(json.dumps(self.d, indent=1), "utf-8")
        except Exception:
            pass
