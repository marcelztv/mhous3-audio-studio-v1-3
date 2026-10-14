"""Headless smoke test of the new interface (offscreen): boot pieces, Auto Clean, keep-or-restart popup,
workspaces, themes, tutorial, batch, screenshots."""
import os, sys, tempfile, time
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"
tmp_home = tempfile.mkdtemp(); os.environ["HOME"] = tmp_home
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import soundfile as sf
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication, QMessageBox
from mhous3 import theme, settings as cfg
from mhous3.config import Config
from mhous3.splash import Splash
from test_engine import messy_clip

OUT = Path(os.environ.get("SHOTS", "/tmp/shots")); OUT.mkdir(exist_ok=True)
app = QApplication([])
def wait(cond, timeout=120):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents(QEventLoop.AllEvents, 40)
    assert cond(), "timeout"

config = Config(); theme.apply(app, "dark")
sp = Splash(); sp.show(); [sp.message(m, f) for m, f in (("Waking the audio engine...", .1), ("Loading signal processing...", .4), ("Preparing the workspaces...", .85))]
for _ in range(25): app.processEvents(); time.sleep(0.03)
sp.grab().save(str(OUT / "splash.png")); g = sp.geometry(); scr = app.primaryScreen().availableGeometry()
print("splash", g.width(), g.height(), "screen", scr.width(), scr.height(), "area ratio %.2f" % (g.width() * g.height() / (scr.width() * scr.height())))
sp.close()

from mhous3.mainwindow import MainWindow
win = MainWindow(config); win.resize(1440, 900); win.show(); app.processEvents()
d = Path(tempfile.mkdtemp()); src = d / "Interview_camera_mic.wav"; sf.write(src, messy_clip(14), 48000)
s = win.session
s.load(str(src)); wait(lambda: s.audio is not None); wait(lambda: s.preview is not None and s.task is None)
win.fast.grab().save(str(OUT / "fast_loaded.png"))

# --- Auto Clean: analyse + apply (dialog bypassed by 'always best')
config.set("always_best", True); win.auto_clean()
wait(lambda: s.cleaned_by is not None); wait(lambda: s.task is None and s.preview is not None)
print("auto clean ->", s.cleaned_by, "NR", win.store.s["nr_amount"], "declip", win.store.s["declip"], "hum", win.store.s["hum"])
win.fast.cards.select(s.cleaned_by); win.store.set("volume_boost", 4.0); wait(lambda: s.task is None)
win.fast.grab().save(str(OUT / "fast_cleaned.png"))

# --- keep-or-restart popup: choose 'Keep' then 'Start from original'
import mhous3.mainwindow as mw
calls = []
def fake_exec(self):
    calls.append(self.text()); self._pick = [b for b in self.buttons() if b.text().startswith("Keep")][0] if len(calls) == 1 else [b for b in self.buttons() if b.text().startswith("Start")][0]
    return 0
QMessageBox.exec = fake_exec; QMessageBox.clickedButton = lambda self: self._pick
win.show_workspace(1); assert win.stack.currentIndex() == 1 and s.cleaned_by and win.store.s["nr_amount"] > 0, "keep failed"
print("popup1 shown:", len(calls) == 1, "| settings kept, NR =", win.store.s["nr_amount"])
s.popup_done = False; win.show_workspace(2); wait(lambda: s.task is None)
assert s.cleaned_by is None and win.store.s["nr_amount"] == 0, "restart failed"
print("popup2 -> restarted from original, NR =", win.store.s["nr_amount"])
win.pick_preset("Cinema Dialogue (Hollywood)"); wait(lambda: s.task is None and s.preview is not None)
win.show_workspace(1, ask=False); app.processEvents()
for r in range(30): app.processEvents(); time.sleep(0.02)
win.prec.grab().save(str(OUT / "precision.png"))
win.show_workspace(2, ask=False); win.full.docks["Pro Controls"].raise_(); app.processEvents(); win.full.grab().save(str(OUT / "full.png"))

# --- playback-free spectrogram scroll check
s.player.pos = 48000; s.player.playing = True; s.tick.emit(); s.player.pos = 48000 * 2; s.tick.emit(); s.player.playing = False
# --- light theme + system
win.set_theme("light"); win.show_workspace(0, ask=False); app.processEvents(); win.grab().save(str(OUT / "fast_light.png"))
win.set_theme("system"); assert theme.current_name() in ("dark", "light"); win.set_theme("dark")
# --- tutorial
def settle(ms):
    t = time.time()
    while time.time() - t < ms / 1000: app.processEvents(QEventLoop.AllEvents, 20)
win.show_workspace(0, ask=False); win.start_tutorial("basic"); settle(900)
for i in range(1, 6): win.tutorial.goto(i); settle(900)
win.grab().save(str(OUT / "tutorial.png")); win.tutorial.end(False)
win.start_tutorial("advanced"); [ (win.tutorial.goto(i), app.processEvents()) for i in range(len(win.tutorial.steps)) ]; win.tutorial.end(True)
assert config.get("tutorial_done") is True
# --- shortcut config persists
cfgd = {"FastTrack": {"play": "P"}}; config.set("shortcuts", cfgd); win.show_workspace(0, ask=False)
assert win.acts["play"].shortcut().toString() == "P", win.acts["play"].shortcut().toString()
# --- batch
win.show_workspace(3, ask=False); win.batch.add_paths([str(src)]); win.batch.same.setChecked(False); win.batch.out_dir = str(d / "out")
win.batch.which.setCurrentText("Podcast / Voice-over"); win.batch.start()
wait(lambda: win.batch.btn_start.isEnabled(), 180)
print("batch:", [win.batch.table.item(0, c).text() for c in (1, 2)]); assert win.batch.table.item(0, 1).text() == "Done"
win.grab().save(str(OUT / "batch.png"))
win.close(); print("GUI OK")
