"""Headless test of the Stage 3 interface: mixer strips, routing, inserts, effect editor, JSFX editor, meters."""
import os, sys, tempfile, time
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"; os.environ["HOME"] = tempfile.mkdtemp()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np, soundfile as sf
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication
from mhous3 import theme, fx, jsfx
from mhous3.config import Config
from test_engine import messy_clip
SR = 48000
OUT = Path(os.environ.get("SHOTS", "/tmp/shots")); OUT.mkdir(exist_ok=True)
app = QApplication([]); cfg = Config(); cfg.set("tutorial_done", True); theme.apply(app, "dark")
def pump(ms):
    t = time.time()
    while time.time() - t < ms / 1000: app.processEvents(QEventLoop.AllEvents, 20)
def wait(c, to=90):
    t = time.time()
    while not c() and time.time() - t < to: app.processEvents(QEventLoop.AllEvents, 30)
    assert c(), "timeout"
from mhous3.mainwindow import MainWindow
from mhous3.mixer import FxEditorDialog, Strip
win = MainWindow(cfg); win.resize(1560, 940); win.show(); pump(100)
d = Path(tempfile.mkdtemp()); a = d / "Dialogue.wav"; b = d / "Music_bed.wav"; c = d / "Foley.wav"
sf.write(a, messy_clip(12), SR)
t = np.arange(SR * 12) / SR; mus = np.stack([0.15 * np.sin(2 * np.pi * 196 * t) + 0.1 * np.sin(2 * np.pi * 294 * t)] * 2, 1).astype(np.float32); sf.write(b, mus, SR)
x = np.zeros(SR * 12, np.float32)
for pos in (1.0, 2.5, 4.0, 6.5, 9.0): i = int(pos * SR); x[i:i + 5000] = 0.6 * np.random.default_rng(int(pos * 7)).standard_normal(5000) * np.hanning(5000)
sf.write(c, np.stack([x, x], 1), SR)
s = win.session; s.load(str(a)); wait(lambda: s.audio is not None)
win.show_workspace(1, ask=False); pump(200); P = win.project; tl = win.prec.timeline
tl.import_files([str(b)], -1, 0); wait(lambda: len(P.tracks) == 2); tl.import_files([str(c)], -1, 0); wait(lambda: len(P.tracks) == 3)
P.tracks[0].name, P.tracks[1].name, P.tracks[2].name = "Dialogue", "Music bed", "Foley"
tl.add_bus(); tl.add_bus(); rev, music = P.tracks[3], P.tracks[4]; rev.name, music.name = "Reverb send", "Music bus"
P.set_route(P.tracks[1].id, music.id); P.set_send(P.tracks[0].id, rev.id, -14, False); P.set_send(P.tracks[2].id, rev.id, -9, True)
P.add_insert(rev.id, "reverb", {"type": "Hall", "mix": 1.0}); P.add_insert(music.id, "comp", {"thr": -24, "ratio": 3})
P.load_track_preset(P.tracks[0].id, "Podcast Host Voice"); v = P.add_vca("Dialogue VCA"); P.tracks[0].vca = v["id"]; v["gain_db"] = -2
P.add_insert(0, "maximizer", {"gain": 3, "ceiling": -1}); P.tracks[1].gain_db = -8
win.prec.tabs.setCurrentIndex(1); pump(400)
mv = win.prec.mixer; kinds = [st.kind for st in mv.strips]; print("strips:", kinds)
assert kinds.count("track") == 3 and kinds.count("bus") == 2 and kinds.count("vca") == 1 and kinds[-1] == "master"
# mix: sends through the reverb bus add a tail; compare with the dry mix
mix = P.mixdown(); assert np.isfinite(mix).all() and len(mix) == 12 * SR
dry = P.mixdown(); P.tracks[0].sends.clear(); P.tracks[2].sends.clear(); assert not np.allclose(mix, P.mixdown(), atol=1e-4); P.set_send(P.tracks[0].id, rev.id, -14, False); P.set_send(P.tracks[2].id, rev.id, -9, True)
# fx editor dialogs
dlg = FxEditorDialog(P, P.tracks[0].id, 1, win); dlg.show(); pump(200); dlg.grab().save(str(OUT / "fx_editor.png")); dlg.close()
P.add_insert(P.tracks[2].id, "jsfx", {}, {"script": jsfx.EXAMPLES["Tremolo"]})
jd = FxEditorDialog(P, P.tracks[2].id, len(P.tracks[2].inserts) - 1, win); jd.show(); pump(200)
jd.edit.setPlainText("desc:Bad\n@sample\nspl0 = nothing(1);"); jd._apply_script(); assert "Script problem" in jd.err.text(); print("JSFX error shown:", jd.err.text())
jd._example(1); jd.edit.setPlainText(jsfx.EXAMPLES["Tremolo"]); jd._apply_script(); assert jd.err.text() == "Script compiled."; pump(400); jd.grab().save(str(OUT / "jsfx_editor.png")); jd.close()
# strip interactions
st = mv.strips[0]; st.fader.setValue(-35); assert abs(P.tracks[0].gain_db + 3.5) < 1e-6
st.mute.setChecked(True); assert P.tracks[0].mute; st.mute.setChecked(False)
st.ins.setCurrentRow(0); st._bypass(); assert P.tracks[0].inserts[0]["bypass"]; st._bypass()
n_ins = len(P.tracks[0].inserts); st.ins.setCurrentRow(0); st._remove(); assert len(P.tracks[0].inserts) == n_ins - 1
P.undo(); P.redo()
# meters panel
s.auto_preview = True; s.request_preview(); wait(lambda: s.preview is not None and s.task is None)
m = win.prec.meters; m.compute(); wait(lambda: bool(m.data)); pump(300); print("meters:", m.lab.text().replace("\n", " | "))
assert "Integrated" in m.lab.text()
win.prec.tabs.setCurrentIndex(1); pump(300); win.grab().save(str(OUT / "mixer.png"))
win.show_workspace(2, ask=False); win.full.tabs.setCurrentIndex(1); pump(400); win.full.docks["Meters"].show(); pump(300); win.grab().save(str(OUT / "mixer_full.png"))
win.close(); print("STAGE3 GUI OK")
