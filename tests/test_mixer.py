import sys, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mhous3 import dsp, fx
from mhous3.project import Project
SR = 48000

def tone(sec=2, f=220, amp=0.4):
    x = (amp * np.sin(2 * np.pi * f * np.arange(int(SR * sec)) / SR)).astype(np.float32)
    return np.stack([x, x], 1)

def rms(x): return float(np.sqrt(np.mean(np.square(x))))

def test_bus_routing_and_master():
    p = Project(); p.import_array("a", tone()); p.import_array("b", tone(2, 330))
    ref = p.mixdown()
    bus = p.add_bus("Music"); p.set_route(p.tracks[0].id, bus.id); p.set_route(p.tracks[1].id, bus.id)
    assert np.allclose(p.mixdown(), ref, atol=1e-5)                       # unity bus = same mix
    bus.gain_db = -6; assert abs(rms(p.mixdown()) / rms(ref) - 0.5012) < 0.01
    bus.gain_db = 0; p.master["gain_db"] = -6; assert abs(rms(p.mixdown()) / rms(ref) - 0.5012) < 0.01
    p.master["gain_db"] = 0; p.master["mute"] = True; assert rms(p.mixdown()) == 0
    p.master["mute"] = False; bus.mute = True; assert rms(p.mixdown()) == 0

def test_sends_pre_post():
    p = Project(); p.import_array("a", tone(2, amp=0.5)); t = p.tracks[0]
    rev = p.add_bus("Reverb"); p.set_route(t.id, 0)
    t.mute = False; t.gain_db = -60                                       # fader down
    p.set_send(t.id, rev.id, db=0.0, pre=False); assert rms(p.mixdown()) < 0.001   # post-fader send follows fader
    p.set_send(t.id, rev.id, db=0.0, pre=True); assert rms(p.mixdown()) > 0.3      # pre-fader send ignores fader
    p.set_send(t.id, rev.id, db=-12.0, pre=True); a = rms(p.mixdown()); p.set_send(t.id, rev.id, db=0.0, pre=True); assert abs(a / rms(p.mixdown()) - 0.251) < 0.01
    p.remove_send(t.id, rev.id); assert rms(p.mixdown()) < 0.001

def test_vca_and_cycles():
    p = Project(); p.import_array("a", tone()); p.import_array("b", tone(2, 330)); base = rms(p.mixdown())
    v = p.add_vca("Dialog"); p.tracks[0].vca = v["id"]; p.tracks[1].vca = v["id"]
    v["gain_db"] = -6; assert abs(rms(p.mixdown()) / base - 0.5012) < 0.01
    v["mute"] = True; assert rms(p.mixdown()) == 0
    q = Project(); q.import_array("a", tone()); b1, b2 = q.add_bus("A"), q.add_bus("B")
    q.set_route(q.tracks[0].id, b1.id); q.set_route(b1.id, b2.id); q.set_route(b2.id, b1.id)      # feedback loop must not hang
    assert np.isfinite(q.mixdown()).all() and rms(q.mixdown()) > 0.1
    q.set_route(b2.id, 0); assert [x.name for x in q.bus_order()] == ["A", "B"]                      # A feeds B so A is processed first

def test_inserts_and_presets():
    p = Project(); p.import_array("a", tone(2, 100)); t = p.tracks[0]
    base = rms(p.mixdown()); p.add_insert(t.id, "hpf", {"f": 1000, "order": 4}); assert rms(p.mixdown()) < base * 0.1
    p.set_insert(t.id, 0, bypass=True); assert abs(rms(p.mixdown()) - base) < 1e-4
    p.set_insert(t.id, 0, bypass=False, params={"f": 20}); assert rms(p.mixdown()) > base * 0.9
    p.add_insert(t.id, "gain", {"db": -6}); p.move_insert(t.id, 1, -1); assert t.inserts[0]["kind"] == "gain"
    p.remove_insert(t.id, 0); assert len(t.inserts) == 1
    p.add_insert(0, "limiter", {"ceiling": -12}); assert np.abs(p.mixdown()).max() <= dsp.db2lin(-12) + 1e-3     # master insert (id 0)
    p.load_track_preset(t.id, "Lead Vocal Chain"); assert len(t.inserts) == 5 and np.isfinite(p.mixdown()).all()
    p.add_insert(t.id, "jsfx", {"slider1": -6.0}, {"script": __import__("mhous3.jsfx", fromlist=["x"]).EXAMPLES["Gain (dB)"]}); assert np.isfinite(p.mixdown()).all()

def test_automation_modes_and_mute_send_auto():
    p = Project(); p.import_array("a", tone(2, amp=0.5)); t = p.tracks[0]
    p.auto_add(t.id, "volume", 0, -60); p.auto_add(t.id, "volume", 2 * SR - 1, -60)
    assert rms(p.mixdown()) < 0.01; t.auto_mode = "Off"; assert rms(p.mixdown()) > 0.3                 # Off ignores the lane
    t.auto_mode = "Read"; t.auto_volume.clear()
    t.auto_extra["mute"] = [(0, 0), (SR, 1)]                                                           # unmuted 0-1 s, muted after
    m = p.mixdown(); assert rms(m[: SR - 500]) > 0.25 and rms(m[SR + 500:]) < 0.01
    t.auto_extra.clear(); bus = p.add_bus("FX"); p.set_send(t.id, bus.id, -80, True); t.mute = False
    t.auto_extra[f"send:{bus.id}"] = [(0, -80), (2 * SR - 1, 0)]; t.output = 0
    base = Project(); base.import_array("a", tone(2, amp=0.5)); assert rms(p.mixdown()[-SR // 2:]) > rms(base.mixdown()[-SR // 2:]) * 1.3   # send ramps up and adds

def test_templates_save_load_stems_undo():
    p = Project(); p.new_from_template("Podcast Recording"); assert [t.name for t in p.tracks][:2] == ["Host", "Guest"] and p.tracks[0].inserts
    p.import_array("a", tone(2), track_id=p.tracks[0].id); bus = p.add_bus("Dialogue"); p.set_route(p.tracks[0].id, bus.id)
    v = p.add_vca("V"); p.tracks[0].vca = v["id"]; v["gain_db"] = -3; p.master["gain_db"] = -1; p.set_send(p.tracks[0].id, bus.id, -9, True)
    d = Path(tempfile.mkdtemp()); out = p.save(d / "s.mh3p"); q = Project(); q.load(out)
    assert np.allclose(q.mixdown(), p.mixdown(), atol=1e-4) and q.vcas == p.vcas and q.master["gain_db"] == -1 and q.tracks[0].inserts[0]["kind"] == "hpf"
    paths = q.render_stems(d / "stems"); names = [x.name for x in paths]; assert any("BUS_Dialogue" in n for n in names); assert len({sf.info(x).frames for x in paths}) == 1
    n_before = len(p.tracks); p.add_bus("X"); assert len(p.tracks) == n_before + 1; p.undo(); assert len(p.tracks) == n_before
    p.add_vca("Y"); p.undo(); assert len(p.vcas) == 1

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
