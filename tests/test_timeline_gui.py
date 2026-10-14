"""Headless test of the multi-track timeline: import, mouse editing (move/trim/fade/gain/automation), operations,
save/load, stems, screenshots."""
import os, sys, tempfile, time
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"; os.environ["HOME"] = tempfile.mkdtemp()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np, soundfile as sf
from PySide6.QtCore import QEventLoop, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from mhous3 import theme
from mhous3.config import Config
from mhous3.timeline import HEADER_W, RULER_H, TRACK_H
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
win = MainWindow(cfg); win.resize(1500, 920); win.show(); pump(100)
d = Path(tempfile.mkdtemp()); a = d / "Interview_A.wav"; b = d / "Foley_hits.wav"
sf.write(a, messy_clip(12), SR)
x = np.zeros(SR * 6, np.float32)
for pos in (0.4, 1.5, 2.6, 4.0):
    i = int(pos * SR); x[i:i + 5000] = 0.7 * np.random.default_rng(2).standard_normal(5000) * np.hanning(5000)
sf.write(b, np.stack([x, x], 1), SR)
s = win.session; s.load(str(a)); wait(lambda: s.audio is not None)
win.show_workspace(1, ask=False); pump(300)
tl = win.prec.timeline; v = tl.view; P = win.project
assert len(P.tracks) == 1 and s.project_mode, "file should appear as Track 1 in project mode"
wait(lambda: abs(len(s.audio) - P.length()) < 2); print("track 1 imported; session audio = mixdown", len(s.audio) / SR, "s")
# import a second file onto a new track at 3 s
tl.import_files([str(b)], -1, 0); wait(lambda: len(P.tracks) == 2); P.move(P.tracks[1].clips[0].id, 3 * SR); P.changed(); pump(200)
v.zoom_fit(); pump(100)
def pt(x, y): return QPoint(int(x), int(y))
tops, _ = v.tops()
c2 = P.tracks[1].clips[0]
# --- mouse: drag clip body right by ~1 s
r = v.clip_rect(P.tracks[1], c2, tops[1]); y = r.center().y() + 10; x0 = r.center().x(); before = c2.start
QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, pt(x0, y)); QTest.mouseMove(v, pt(x0 + v.pps, y)); QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, pt(x0 + v.pps, y))
assert abs((c2.start - before) / SR - 1.0) < 0.05, (c2.start - before) / SR
print("dragged clip +%.2fs" % ((c2.start - before) / SR))
# --- mouse: trim right edge left by 1 s
r = v.clip_rect(P.tracks[1], c2, tops[1]); L = c2.length
QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, pt(r.right() - 1, y)); QTest.mouseMove(v, pt(r.right() - 1 - v.pps, y)); QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, pt(r.right() - 1 - v.pps, y))
assert 0.9 < (L - c2.length) / SR < 1.1, (L - c2.length) / SR
print("trimmed %.2fs off the end" % ((L - c2.length) / SR))
# --- mouse: fade-in handle drag
r = v.clip_rect(P.tracks[1], c2, tops[1])
QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, pt(r.left() + 1, r.top() + 4)); QTest.mouseMove(v, pt(r.left() + 0.5 * v.pps, r.top() + 4)); QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, pt(r.left() + 0.5 * v.pps, r.top() + 4))
print("fade-in set to %.2fs" % (c2.fade_in / SR)); assert 0.4 < c2.fade_in / SR < 0.6
# --- mouse: clip gain line drag (down 10 dB-ish)
r = v.clip_rect(P.tracks[1], c2, tops[1]); gy = r.top() + (12 - c2.gain_db) / 36 * r.height()
QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, pt(r.center().x(), gy)); QTest.mouseMove(v, pt(r.center().x(), gy + r.height() * 10 / 36)); QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, pt(r.center().x(), gy + r.height() * 10 / 36))
print("clip gain now %.1f dB" % c2.gain_db); assert -11 < c2.gain_db < -9
# --- track header: mute toggle, then back; fader drag
t1 = P.tracks[0]; QTest.mouseClick(v, Qt.LeftButton, Qt.NoModifier, pt(14 + 34 + 10, tops[0] + 40)); assert t1.solo
QTest.mouseClick(v, Qt.LeftButton, Qt.NoModifier, pt(14 + 34 + 10, tops[0] + 40)); assert not t1.solo
QTest.mouseClick(v, Qt.LeftButton, Qt.NoModifier, pt(14 + 68 + 10, tops[0] + 40)); assert t1.show_auto
# --- automation: click in lane to add points, drag one
tops, _ = v.tops(); aty = tops[0] + TRACK_H + 30
for sec in (1.0, 4.0, 8.0):
    QTest.mouseClick(v, Qt.LeftButton, Qt.NoModifier, pt(v.sx(sec * SR), aty))
print("automation points:", len(t1.auto_volume)); assert len(t1.auto_volume) == 3
# --- operations
ids = [P.tracks[0].clips[0].id]; v.sel_clips = set(ids); v.sel_track = t1.id
tl.sep_transients(); n_tr = len(P.tracks[0].clips); print("separate at transients -> clips on track 1:", n_tr)
v.sel_clips = {P.tracks[1].clips[0].id}; tl.duplicate(); tl.repeat.__self__
P.undo(); P.redo()
P.set_clip([P.tracks[1].clips[0].id], pitch=3.0); mix = P.mixdown(); assert np.isfinite(mix).all()
# --- A/B + cleaned preview of the mix
s.auto_preview = True; s.request_preview(); wait(lambda: s.preview is not None and s.task is None)
print("master cleanup on mix:", {k: round(v_, 1) for k, v_ in s.preview[1].items() if k in ("lufs_in", "lufs_out")})
v.zoom_fit(); v.sel_clips = {P.tracks[1].clips[0].id}; v.sel_track = P.tracks[1].id; v.update(); pump(300)
win.grab().save(str(OUT / "timeline.png"))
# --- save / load / stems
proj = d / "show" / "My Show.mh3p"; P.save(proj); a.unlink(); b.unlink()
from mhous3.project import Project
Q = Project(); Q.load(proj); assert len(Q.tracks) == 2 and np.allclose(Q.mixdown(), P.mixdown(), atol=1e-4); print("project reloaded OK after deleting originals")
paths = Q.render_stems(d / "stems"); print("stems:", [p.name for p in paths])
win.show_workspace(2, ask=False); pump(300); win.grab().save(str(OUT / "timeline_full.png"))
win.close(); print("TIMELINE GUI OK")
