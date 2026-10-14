# PyInstaller spec - builds dist/MHous3 Audio Studio/ (ffmpeg + DeepFilterNet go inside via bin/)
from PyInstaller.utils.hooks import collect_all

datas = [("assets", "assets"), ("bin", "bin")]
binaries, hidden = [], []
for pkg in ("sounddevice", "soundfile"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hidden += h

a = Analysis(["main.py"], pathex=["."], binaries=binaries, datas=datas, hiddenimports=hidden,
             excludes=["tkinter", "matplotlib", "PyQt5", "PyQt6", "IPython", "pytest", "PySide6.QtWebEngineCore",
                       "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="MHous3 Audio Studio", console=False,
          icon="assets/icon.ico", upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="MHous3 Audio Studio")
