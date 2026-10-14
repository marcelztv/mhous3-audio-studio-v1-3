import sys, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from mhous3 import dsp
from mhous3.project import Project
SR = 48000

def tone(sec, f=220, amp=0.5):
    t = np.arange(int(SR * sec)) / SR
    x = (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)
    return np.stack([x, x], 1)

def test_mix_gain_pan_mute_solo():
    p = Project(); a = p.import_array("a.wav", tone(2)); b = p.import_array("b.wav", tone(2, 440))
    assert len(p.tracks) == 2 and p.length() == 2 * SR
    mix = p.mixdown(); ref = p.render_track(p.tracks[0]) + p.render_track(p.tracks[1])
    assert np.allclose(mix, ref, atol=1e-5)
    p.tracks[0].gain_db = -6; assert abs(np.abs(p.render_track(p.tracks[0])).max() - 0.5 * 10 ** (-6 / 20)) < 1e-3
    p.tracks[0].pan = 1.0; r = p.render_track(p.tracks[0]); assert np.abs(r[:, 0]).max() < 1e-6 and np.abs(r[:, 1]).max() > 0.2
    p.tracks[0].pan = 0; p.tracks[1].mute = True; assert np.allclose(p.mixdown(), p.render_track(p.tracks[0], apply_mute=False), atol=1e-5)
    p.tracks[1].mute = False; p.tracks[0].solo = True; assert np.allclose(p.mixdown(), p.render_track(p.tracks[0]), atol=1e-5)

def test_fades_and_crossfade():
    n = 1000
    fi, fo = dsp.fade_curve(n, "equal_power", True), dsp.fade_curve(n, "equal_power", False)
    assert np.allclose(fi ** 2 + fo ** 2, 1, atol=1e-4) and abs(fi[n // 2] - 0.707) < 0.01
    p = Project(); c = p.import_array("a.wav", tone(1)); p.set_fades([c.id], fade_in=SR // 4, fade_out=SR // 4)
    r = p.render_track(p.tracks[0]); assert abs(r[0]).max() < 1e-3 and np.abs(r[SR // 2 - 300:SR // 2 + 300]).max() > 0.45
    # overlap -> automatic crossfade
    t = p.tracks[0]; s2 = p.add_source("b", tone(1, 330)); c2 = p.add_clip(t.id, s2.id, start=SR - SR // 5)
    eff = p.effective_fades(t); assert eff[c.id][1] >= SR // 5 and eff[c2.id][0] >= SR // 5

def test_edit_operations():
    p = Project(); c = p.import_array("a.wav", tone(4)); tid = p.tracks[0].id
    new = p.split([c.id], 2 * SR); assert len(p.tracks[0].clips) == 2 and p.find_clip(new[0])[1].offset == 2 * SR
    p.undo(); assert len(p.tracks[0].clips) == 1; p.redo(); assert len(p.tracks[0].clips) == 2
    r = p.find_clip(new[0])[1]; p.set_clip([r.id], lock=True); p.split([r.id], 3 * SR); assert len(p.tracks[0].clips) == 2  # locked: no split
    p.set_clip([r.id], lock=False)
    d = p.duplicate([c.id]); assert len(p.tracks[0].clips) == 3
    rp = p.repeat([c.id], 2); assert len(p.tracks[0].clips) == 5
    p.copy_clips([c.id]); got = p.paste(0, 10 * SR); assert p.find_clip(got[0])[1].start == 10 * SR
    before = p.find_clip(c.id)[1].start; p.nudge([c.id], 480); assert p.find_clip(c.id)[1].start == before + 480
    p.delete([c.id]); assert p.find_clip(c.id)[1] is None
    # shuffle closes gaps
    q = Project(); a = q.import_array("a.wav", tone(1)); b = q.add_clip(q.tracks[0].id, a.source_id, start=3 * SR); q.delete([a.id], shuffle=True)
    assert q.find_clip(b.id)[1].start == 3 * SR      # only clip left stays where it was
    q2 = Project(); x = q2.import_array("a.wav", tone(1)); y = q2.add_clip(q2.tracks[0].id, x.source_id, start=2 * SR); z = q2.add_clip(q2.tracks[0].id, x.source_id, start=4 * SR)
    q2.delete([y.id], shuffle=True); assert q2.find_clip(z.id)[1].start == SR
    # trims
    w = Project(); k = w.import_array("a.wav", tone(4)); w.trim_start(k.id, SR); w.trim_end(k.id, 3 * SR)
    assert k.start == SR and k.offset == SR and k.length == 2 * SR
    w.trim_end(k.id, 20 * SR); assert k.length == 3 * SR       # cannot exceed source without loop
    w.trim_end(k.id, 20 * SR, allow_loop=True); assert k.loop and k.length == 19 * SR and len(w.clip_audio(k)) == 19 * SR

def test_grid_and_transients():
    p = Project(); c = p.import_array("a.wav", tone(3)); p.separate_at_grid([c.id], 1.0); assert len(p.tracks[0].clips) == 3
    x = np.zeros(SR * 4, np.float32)
    for pos in (0.5, 1.5, 2.6, 3.4):
        i = int(pos * SR); x[i:i + 4000] = 0.6 * np.random.default_rng(1).standard_normal(4000)
    q = Project(); k = q.import_array("hits.wav", np.stack([x, x], 1)); n = q.separate_at_transients([k.id]); assert n == 4, n

def test_automation_and_fx():
    p = Project(); c = p.import_array("a.wav", tone(2, amp=0.5)); t = p.tracks[0]
    p.auto_add(t.id, "volume", 0, -60); p.auto_add(t.id, "volume", 2 * SR - 1, 0)
    r = p.render_track(t); assert np.abs(r[: SR // 10]).max() < 0.01 and np.abs(r[-SR // 10:]).max() > 0.4      # fades up
    t.auto_volume.clear(); t.lowcut = 2000; r2 = p.render_track(t); assert np.abs(r2[SR:]).max() < 0.2          # 220 Hz removed
    t.lowcut = 0; t.eq_high = 6; assert np.isfinite(p.render_track(t)).all()

def test_pitch_shift():
    x = tone(1.5, 440)[:, :1]; y = dsp.pitch_shift(x, 12)
    assert len(y) == len(x)
    spec = np.abs(np.fft.rfft(y[:, 0] * np.hanning(len(y)))); f = np.argmax(spec) * SR / len(y)
    assert abs(f - 880) < 25, f
    z = dsp.pitch_shift(x, -5); f2 = np.argmax(np.abs(np.fft.rfft(z[:, 0] * np.hanning(len(z))))) * SR / len(z); assert abs(f2 - 440 * 2 ** (-5 / 12)) < 20, f2

def test_save_load_consolidation_and_stems():
    d = Path(tempfile.mkdtemp()); orig = d / "raw.wav"; sf.write(orig, tone(2), SR)
    data, _ = sf.read(orig, dtype="float32", always_2d=True)
    p = Project(); p.import_array("raw.wav", data, str(orig)); p.import_array("b.wav", tone(1, 330), "x"); p.set_fades([p.tracks[0].clips[0].id], fade_in=2000)
    p.tracks[1].auto_pan.append((0, -0.5)); p.tracks[0].gain_db = -3
    out = p.save(d / "proj" / "My Show.mh3p"); orig.unlink()                       # original renamed/deleted -> project must still load
    q = Project(); q.load(out)
    assert len(q.tracks) == 2 and np.allclose(q.mixdown(), p.mixdown(), atol=1e-5) and q.tracks[1].auto_pan == [(0, -0.5)]
    paths = q.render_stems(d / "stems"); assert len(paths) == 2
    lens = {sf.info(x).frames for x in paths}; assert len(lens) == 1 and lens.pop() == q.length()
    assert any((d / "proj" / "My Show_media").iterdir())

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
