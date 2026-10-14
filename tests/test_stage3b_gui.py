"""Headless test: macros, session settings (timecode / bars|beats), musical grid, session export/import, templates."""
import os, sys, tempfile, time
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"; os.environ["HOME"] = tempfile.mkdtemp()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np, soundfile as sf
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication, QMessageBox
QMessageBox.information = staticmethod(lambda *a, **k: print('  [dialog]', str(a[2])[:70].replace(chr(10), ' ')))
QMessageBox.warning = staticmethod(lambda *a, **k: print('  [warning]', str(a[2])[:90].replace(chr(10), ' ')))
from mhous3 import theme, interchange as ix
from mhous3.config import Config
from test_engine import messy_clip
SR = 48000; OUT = Path(os.environ.get("SHOTS", "/tmp/shots")); OUT.mkdir(exist_ok=True)
app = QApplication([]); cfg = Config(); cfg.set("tutorial_done", True); theme.apply(app, "dark")
def pump(ms):
    t = time.time()
    while time.time() - t < ms / 1000: app.processEvents(QEventLoop.AllEvents, 20)
def wait(c, to=90):
    t = time.time()
    while not c() and time.time() - t < to: app.processEvents(QEventLoop.AllEvents, 30)
    assert c(), "timeout"
from mhous3.mainwindow import MainWindow, SessionDialog
from mhous3.macros import MacroDialog, run_macro
win = MainWindow(cfg); win.resize(1500, 920); win.show(); pump(100)
d = Path(tempfile.mkdtemp()); a = d / "Dialogue.wav"; sf.write(a, messy_clip(12), SR)
s = win.session; s.load(str(a)); wait(lambda: s.audio is not None); win.show_workspace(1, ask=False); pump(200)
P, tl = win.project, win.prec.timeline
# ---- macros
cfg.set("macros", {"Chop and fade": ["select_all", "sep_transients", "select_all", "fade20", "zoom_fit"]}); win.refresh_macro_menu()
n0 = len(P.tracks[0].clips); done = run_macro(win, cfg.get("macros")["Chop and fade"]); n1 = len(P.tracks[0].clips)
print(f"macro ran {done} steps: clips {n0} -> {n1}, fade of first clip = {P.tracks[0].clips[0].fade_in / SR * 1000:.0f} ms")
assert done == 5 and n1 > n0 and all(abs(c.fade_in - int(0.02 * SR)) < 2 for c in P.tracks[0].clips)
P.undo(); P.undo(); P.redo()
md = MacroDialog(win); md.resize(700, 460); md.show(); pump(200); md.grab().save(str(OUT / "macros.png")); md.close()
# ---- session settings
st = P.settings; st.update(ruler="Timecode", fps="25"); lab = tl.view._ruler_label(2.0, 1); assert lab == "00:00:02:00", lab
st.update(ruler="Bars|Beats", tempo=90.0, sig_num=4); lab = tl.view._ruler_label(2.0, 1); assert lab == "1|4|000", lab
tl.grid.setCurrentText("1/8 note"); assert abs(tl.view.grid - 60 / 90 / 2) < 1e-9; tl.grid.setCurrentText("1 bar"); assert abs(tl.view.grid - 60 / 90 * 4) < 1e-9
dlg = SessionDialog(win); dlg.buf.setCurrentText("512"); dlg._ok(); assert cfg.get("buffer") == 512 and s.player.blocksize == 512
tl.grid.setCurrentText("1/4 note"); tl.view.zoom_fit(); tl.view.update(); pump(300); win.grab().save(str(OUT / "timeline_bbt.png"))
dlg2 = SessionDialog(win); dlg2.show(); pump(200); dlg2.grab().save(str(OUT / "session_settings.png")); dlg2.close()
st.update(ruler="Min:Sec")
# ---- session export / import
for kind, name in (("rpp", "x.rpp"), ("aaf", "x.aaf"), ("fcpxml", "x.xml")):
    win.export_session(kind, str(d / name)); assert (d / name).exists(), kind
win.export_session("edl", str(d / "edl")); assert list((d / "edl").glob("*.edl"))
before = P.mixdown(); win.import_session("rpp", str(d / "x.rpp")); assert len(P.tracks) == 1 and np.allclose(P.mixdown(), before, atol=2e-3)
win.import_session("fcpxml", str(d / "x.xml")); assert len(P.tracks) == 1 and len(P.tracks[0].clips) == n1
print("session formats exported and re-imported OK")
# ---- template
win.new_from_template("Podcast Recording"); assert [t.name for t in P.tracks] == ["Host", "Guest", "Music bed", "SFX"] and P.tracks[0].inserts
# ---- export options reach the file
win.store.set("out_sr", 44100); win.store.set("out_bits", 16)
win.session.load(str(a)); wait(lambda: s.audio is not None and not s.project_mode)
dst = d / "final.wav"; s.export_to(str(dst)); wait(lambda: dst.exists() and not s.task, 120); pump(500)
i = sf.info(dst); print("export:", i.samplerate, "Hz,", i.subtype); assert i.samplerate == 44100 and i.subtype == "PCM_16"
win.close(); print("STAGE3B GUI OK")
