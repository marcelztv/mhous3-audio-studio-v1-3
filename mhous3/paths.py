"""Locate bundled tools (ffmpeg, deep-filter) both in dev and in the frozen installer build."""
import shutil
import subprocess
import sys
from pathlib import Path


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def find_tool(name: str):
    for d in (app_dir() / "bin", app_dir(), Path(sys.executable).parent / "bin"):
        for n in (name + ".exe", name):
            p = d / n
            if p.exists():
                return str(p)
    return shutil.which(name)


NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
