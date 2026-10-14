import sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mhous3 import interchange as ix
from mhous3.project import Project
SR = 48000

def tone(sec, f=220, amp=0.4):
    x = (amp * np.sin(2 * np.pi * f * np.arange(int(SR * sec)) / SR)).astype(np.float32)
    return np.stack([x, x * 0.8], 1)

def make():
    p = Project(); p.name = "My Show"
    a = p.import_array("Host.wav", tone(6)); b = p.import_array("Music.wav", tone(5, 330))
    p.tracks[0].name, p.tracks[1].name = "Host", "Music bed"
    p.split([a.id], 2 * SR); p.move(b.id, SR); p.set_fades([b.id], fade_in=SR // 4, fade_out=SR // 2, in_curve="linear")
    p.set_clip([b.id], gain_db=-6.0); p.tracks[1].gain_db = -3.0; p.tracks[1].pan = 0.25; p.tracks[0].solo = False
    p.tracks[0].clips[1].mute = True
    return p

def test_timecode():
    assert ix.frames_to_tc(90000, "25") == "01:00:00:00" and ix.frames_to_tc(25 * 61 + 3, "25") == "00:01:01:03"
    assert ix.frames_to_tc(1800, "29.97 DF") == "00:01:00;02" and ix.frames_to_tc(17982, "29.97 DF") == "00:10:00;00"
    for key in ix.FPS:
        for fr in (0, 1, 1799, 1800, 17981, 17982, 107892, 250000):
            assert ix.tc_to_frames(ix.frames_to_tc(fr, key), key) == fr, (key, fr)
    assert ix.sec_to_frames(10, "23.976") == 240 and ix.sec_to_frames(1, "24") == 24

def test_rpp_roundtrip():
    p = make(); d = Path(tempfile.mkdtemp()); out = ix.export_rpp(p, d / "show.rpp")
    txt = out.read_text(); assert "<TRACK" in txt and 'NAME "Music bed"' in txt and "<SOURCE WAVE" in txt
    q = Project(); ix.import_rpp(q, out)
    assert [t.name for t in q.tracks] == ["Host", "Music bed"]
    for t1, t2 in zip(p.tracks, q.tracks):
        assert len(t1.clips) == len(t2.clips) and abs(t1.gain_db - t2.gain_db) < 0.05 and abs(t1.pan - t2.pan) < 1e-3
        for c1, c2 in zip(sorted(t1.clips, key=lambda c: c.start), sorted(t2.clips, key=lambda c: c.start)):
            assert (c1.start, c1.offset, c1.length, c1.mute) == (c2.start, c2.offset, c2.length, c2.mute)
            assert abs(c1.gain_db - c2.gain_db) < 0.05 and abs(c1.fade_in - c2.fade_in) < 2 and abs(c1.fade_out - c2.fade_out) < 2
    assert np.allclose(p.mixdown(), q.mixdown(), atol=2e-3)
    print("  RPP round trip: identical clips, fades, gain and mix")

def test_edl():
    p = make(); d = Path(tempfile.mkdtemp()); files = ix.export_edl(p, d, "25", start_hour=1)
    assert len(files) == 2
    txt = files[1].read_text(); assert "FCM: NON-DROP FRAME" in txt and "* FROM CLIP NAME: Music.wav" in txt
    ev = [l for l in txt.splitlines() if l.startswith("001")][0].split()
    assert ev[-4:] == ["00:00:00:00", "00:00:05:00", "01:00:01:00", "01:00:06:00"], ev          # moved to 1 s, 5 s long
    df = ix.export_edl(p, d / "df", "29.97 DF")[0].read_text(); assert "FCM: DROP FRAME" in df and ";" in df
    print("  EDL events:", ev[-4:])

def test_fcpxml_roundtrip():
    p = make(); d = Path(tempfile.mkdtemp()); out = ix.export_fcpxml(p, d / "show.xml", "25")
    import xml.etree.ElementTree as ET
    root = ET.fromstring(out.read_text().replace("<!DOCTYPE xmeml>", ""))
    assert root.tag == "xmeml" and len(root.findall(".//clipitem")) == 3 and len(root.findall(".//track")) == 2
    q = Project(); ix.import_fcpxml(q, out); assert [len(t.clips) for t in q.tracks] == [2, 1]
    c1 = sorted(p.tracks[0].clips, key=lambda c: c.start)[1]; c2 = sorted(q.tracks[0].clips, key=lambda c: c.start)[1]
    assert (c1.start, c1.offset, c1.length) == (c2.start, c2.offset, c2.length) and c2.mute
    print("  FCP XML round trip OK")

def test_aaf():
    import aaf2
    p = make(); d = Path(tempfile.mkdtemp()); out = ix.export_aaf(p, d / "show.aaf"); assert out.exists() and out.stat().st_size > 10000
    with aaf2.open(str(out), "r") as f:
        comps = list(f.content.compositionmobs()); assert len(comps) == 1 and comps[0].name == "My Show"
        slots = list(comps[0].slots); assert len(slots) == 2
        kinds = [[type(c).__name__ for c in s.segment.components] for s in slots]
        assert kinds[1] == ["Filler", "SourceClip"], kinds                           # music starts after a 1 s gap
        assert slots[1].segment.components[0].length == SR and slots[1].segment.components[1].length == 5 * SR
        assert kinds[0] == ["SourceClip"]                                           # the muted second Host clip is skipped
        assert len(list(f.content.mastermobs())) >= 2
    print("  AAF: composition with", len(slots), "tracks, embedded masters, gap fillers")

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
