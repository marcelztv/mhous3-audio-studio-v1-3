"""Session interchange.  Carried across: tracks, clip positions/offsets/lengths, names, clip gain, fades, mute, track
fader/pan/mute/solo (RPP).  NOT carried: inserts, sends/buses, automation, pitch (those are specific to this program)."""
import re
import shlex
import xml.etree.ElementTree as ET
from fractions import Fraction
from math import gcd
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np
import soundfile as sf
from scipy import signal

from . import dsp
from .dsp import SR
from .project import Clip, Project

FPS = {  # label: (timebase, ntsc, drop-frame)
    "23.976": (24, True, False), "24": (24, False, False), "25": (25, False, False),
    "29.97 DF": (30, True, True), "29.97 NDF": (30, True, False), "30": (30, False, False),
}


# ============================================================================ timecode
def real_fps(key):
    base, ntsc, _ = FPS[key]
    return base * 1000 / 1001 if ntsc else float(base)


def sec_to_frames(sec, key):
    return int(round(sec * real_fps(key)))


def frames_to_tc(fn, key):
    base, _, drop = FPS[key]
    fn = int(fn)
    if drop:
        d, m = divmod(fn, 17982)
        fn += 18 * d + (2 * ((m - 2) // 1798) if m > 1 else 0)
    ff, ss, mm, hh = fn % base, (fn // base) % 60, (fn // (base * 60)) % 60, fn // (base * 3600)
    return f"{hh:02d}:{mm:02d}:{ss:02d}{';' if drop else ':'}{ff:02d}"


def tc_to_frames(tc, key):
    base, _, drop = FPS[key]
    hh, mm, ss, ff = [int(x) for x in re.split(r"[:;]", tc)]
    total = ((hh * 60 + mm) * 60 + ss) * base + ff
    if drop:
        tmin = hh * 60 + mm
        total -= 2 * (tmin - tmin // 10)
    return total


def _frames(samples, key):
    return sec_to_frames(samples / SR, key)


# ============================================================================ media helpers
def load_audio(path):
    """Any audio/video file -> float32 (n, 2) at 48 kHz."""
    path = str(path)
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        data = data[:, :2]
        if sr != SR:
            g = gcd(sr, SR)
            data = signal.resample_poly(data, SR // g, sr // g, axis=0).astype(np.float32)
        return data
    except Exception:
        from . import engine
        return engine.decode(path)


def _consolidate(project, folder):
    """Write every source used by the project into `folder` (so exported sessions never point at moved files)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for sid in {c.source_id for t in project.tracks for c in t.clips}:
        p = folder / f"source_{sid}.wav"
        if not p.exists():
            sf.write(p, project.sources[sid].data, SR, subtype="PCM_24")
        paths[sid] = p
    return paths


def _audio_tracks(project):
    return [t for t in project.tracks if not t.is_bus]


# ============================================================================ REAPER RPP
def export_rpp(project, path):
    path = Path(path)
    media = _consolidate(project, path.with_name(path.stem + "_media"))
    L = ['<REAPER_PROJECT 0.1 "6.80/win64" 1700000000', "  RIPPLE 0", f"  SAMPLERATE {SR} 0 0",
         f"  TEMPO {project.settings.get('tempo', 120)} {project.settings.get('sig_num', 4)} {project.settings.get('sig_den', 4)}"]
    for t in _audio_tracks(project):
        L += ["  <TRACK", f'    NAME "{t.name}"', f"    VOLPAN {dsp.db2lin(t.gain_db):.6f} {t.pan:.4f} -1 -1 1",
              f"    MUTESOLO {int(t.mute)} {int(t.solo)} 0"]
        for c in sorted(t.clips, key=lambda c: c.start):
            fi, fo = c.fade_in / SR, c.fade_out / SR
            shape = lambda k: 1 if k == "equal_power" else 0
            L += ["    <ITEM", f"      POSITION {c.start / SR:.9f}", "      SNAPOFFS 0", f"      LENGTH {c.length / SR:.9f}", f"      LOOP {int(c.loop)}",
                  f"      MUTE {int(c.mute)} 0", f"      FADEIN {shape(c.fade_in_curve)} {fi:.6f} 0 1 0 0 0", f"      FADEOUT {shape(c.fade_out_curve)} {fo:.6f} 0 1 0 0 0",
                  f"      VOLPAN {dsp.db2lin(c.gain_db):.6f} 0 1 -1", f"      SOFFS {c.offset / SR:.9f}", f'      NAME "{c.name}"',
                  "      <SOURCE WAVE", f'        FILE "{media[c.source_id].parent.name}/{media[c.source_id].name}"', "      >", "    >"]
        L.append("  >")
    L.append(">")
    path.write_text("\n".join(L) + "\n", "utf-8")
    return path


def _parse_blocks(text):
    """REAPER's nested <BLOCK ... > text format -> (name, args, lines, children)."""
    root = ("ROOT", [], [], [])
    stack = [root]
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("<"):
            parts = shlex.split(line[1:], posix=True)
            node = (parts[0], parts[1:], [], [])
            stack[-1][3].append(node)
            stack.append(node)
        elif line == ">":
            if len(stack) > 1:
                stack.pop()
        else:
            try:
                parts = shlex.split(line, posix=True)
            except ValueError:
                parts = line.split()
            stack[-1][2].append(parts)
    return root


def import_rpp(project, path):
    path = Path(path)
    root = _parse_blocks(path.read_text("utf-8", errors="replace"))
    proj = root[3][0] if root[3] else root
    project.clear()
    cache = {}
    for node in proj[3]:
        if node[0] != "TRACK":
            continue
        kv = {ln[0]: ln[1:] for ln in node[2] if ln}
        t = project.add_track(" ".join(kv.get("NAME", ["Track"])), push=False)
        if "VOLPAN" in kv:
            t.gain_db = round(float(20 * np.log10(max(float(kv["VOLPAN"][0]), 1e-6))), 2)
            t.pan = float(kv["VOLPAN"][1])
        if "MUTESOLO" in kv:
            t.mute, t.solo = bool(int(kv["MUTESOLO"][0])), bool(int(kv["MUTESOLO"][1]))
        for item in node[3]:
            if item[0] != "ITEM":
                continue
            ik = {ln[0]: ln[1:] for ln in item[2] if ln}
            src_node = next((c for c in item[3] if c[0] == "SOURCE"), None)
            if not src_node:
                continue
            fk = next((ln[1] for ln in src_node[2] if ln and ln[0] == "FILE"), None)
            if not fk:
                continue
            fp = Path(fk) if Path(fk).is_absolute() else path.parent / fk
            if not fp.exists():
                continue
            if str(fp) not in cache:
                cache[str(fp)] = project.add_source(fp.name, load_audio(fp), str(fp))
            src = cache[str(fp)]
            c = project.add_clip(t.id, src.id, int(float(ik["POSITION"][0]) * SR), int(float(ik.get("SOFFS", ["0"])[0]) * SR),
                                 int(float(ik["LENGTH"][0]) * SR), " ".join(ik.get("NAME", [fp.stem])), push=False)
            if "VOLPAN" in ik:
                c.gain_db = round(float(20 * np.log10(max(float(ik["VOLPAN"][0]), 1e-6))), 2)
            c.mute = "MUTE" in ik and ik["MUTE"][0] == "1"
            c.loop = "LOOP" in ik and ik["LOOP"][0] == "1"
            if "FADEIN" in ik:
                c.fade_in = int(float(ik["FADEIN"][1]) * SR)
                c.fade_in_curve = "equal_power" if ik["FADEIN"][0] == "1" else "linear"
            if "FADEOUT" in ik:
                c.fade_out = int(float(ik["FADEOUT"][1]) * SR)
                c.fade_out_curve = "equal_power" if ik["FADEOUT"][0] == "1" else "linear"
    project.name = path.stem
    project.changed()


# ============================================================================ CMX3600 EDL
def export_edl(project, folder, fps="25", start_hour=1):
    """One EDL per audio track (the CMX3600 format is single-track).  Record timecode starts at <start_hour>:00:00:00."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base_frames = start_hour * 3600 * FPS[fps][0]
    out = []
    for i, t in enumerate(_audio_tracks(project), 1):
        lines = [f"TITLE: {project.name} - {t.name}", f"FCM: {'DROP FRAME' if FPS[fps][2] else 'NON-DROP FRAME'}", ""]
        for n, c in enumerate(sorted(t.clips, key=lambda c: c.start), 1):
            s_in, s_out = _frames(c.offset, fps), _frames(c.offset + c.length, fps)
            r_in, r_out = base_frames + _frames(c.start, fps), base_frames + _frames(c.end, fps)
            reel = (re.sub(r"[^A-Za-z0-9]", "", Path(c.name).stem) or "AX")[:8].upper()
            lines.append(f"{n:03d}  {reel:<8s} AA       C        {frames_to_tc(s_in, fps)} {frames_to_tc(s_out, fps)} "
                         f"{frames_to_tc(r_in, fps)} {frames_to_tc(r_out, fps)}")
            lines.append(f"* FROM CLIP NAME: {c.name}")
            if c.gain_db:
                lines.append(f"* CLIP GAIN: {c.gain_db:+.1f} dB")
        p = folder / f"{i:02d}_{re.sub(r'[^A-Za-z0-9_-]', '_', t.name)}.edl"
        p.write_text("\n".join(lines) + "\n", "utf-8")
        out.append(p)
    return out


# ============================================================================ Final Cut Pro XML (xmeml v4)
def _rate(parent, fps):
    base, ntsc, _ = FPS[fps]
    r = ET.SubElement(parent, "rate")
    ET.SubElement(r, "timebase").text = str(base)
    ET.SubElement(r, "ntsc").text = "TRUE" if ntsc else "FALSE"


def export_fcpxml(project, path, fps="25"):
    path = Path(path)
    media = _consolidate(project, path.with_name(path.stem + "_media"))
    root = ET.Element("xmeml", version="4")
    seq = ET.SubElement(root, "sequence", id="sequence-1")
    ET.SubElement(seq, "name").text = project.name
    ET.SubElement(seq, "duration").text = str(_frames(project.length(), fps))
    _rate(seq, fps)
    audio = ET.SubElement(ET.SubElement(seq, "media"), "audio")
    ET.SubElement(audio, "numOutputChannels").text = "2"
    n = 0
    files = {}
    for ti, t in enumerate(_audio_tracks(project), 1):
        tr = ET.SubElement(audio, "track")
        for c in sorted(t.clips, key=lambda c: c.start):
            n += 1
            ci = ET.SubElement(tr, "clipitem", id=f"clipitem-{n}")
            ET.SubElement(ci, "name").text = c.name
            ET.SubElement(ci, "enabled").text = "FALSE" if c.mute else "TRUE"
            ET.SubElement(ci, "duration").text = str(_frames(len(project.sources[c.source_id].data), fps))
            _rate(ci, fps)
            ET.SubElement(ci, "start").text = str(_frames(c.start, fps))
            ET.SubElement(ci, "end").text = str(_frames(c.end, fps))
            ET.SubElement(ci, "in").text = str(_frames(c.offset, fps))
            ET.SubElement(ci, "out").text = str(_frames(c.offset + c.length, fps))
            fid = files.setdefault(c.source_id, f"file-{len(files) + 1}")
            f = ET.SubElement(ci, "file", id=fid)
            ET.SubElement(f, "name").text = media[c.source_id].name
            ET.SubElement(f, "pathurl").text = "file://localhost/" + quote(str(media[c.source_id].resolve()).replace("\\", "/").lstrip("/"))
            _rate(f, fps)
            ET.SubElement(f, "duration").text = str(_frames(len(project.sources[c.source_id].data), fps))
            ET.SubElement(ET.SubElement(ET.SubElement(f, "media"), "audio"), "channelcount").text = "2"
            st = ET.SubElement(ci, "sourcetrack")
            ET.SubElement(st, "mediatype").text = "audio"
            ET.SubElement(st, "trackindex").text = str(ti)
            # sample-accurate placement (frames are rounded above)
            lab = ET.SubElement(ci, "labels")
            ET.SubElement(lab, "label2").text = f"samples start={c.start} offset={c.offset} length={c.length} gain={c.gain_db}"
    ET.indent(root)
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + ET.tostring(root, encoding="unicode"), "utf-8")
    return path


def import_fcpxml(project, path):
    path = Path(path)
    root = ET.fromstring(re.sub(r"<!DOCTYPE[^>]*>", "", path.read_text("utf-8")))
    seq = root.find("sequence")
    base = int(seq.findtext("rate/timebase", "25"))
    ntsc = seq.findtext("rate/ntsc", "FALSE").upper() == "TRUE"
    fps = base * 1000 / 1001 if ntsc else float(base)
    project.clear()
    cache = {}
    for tr in seq.findall("media/audio/track"):
        t = project.add_track(push=False)
        for ci in tr.findall("clipitem"):
            url = ci.findtext("file/pathurl", "")
            fp = Path(unquote(url.replace("file://localhost/", "").replace("file:///", "/")))
            if not fp.is_absolute() and not fp.exists():
                fp = Path("/" + str(fp)) if Path("/" + str(fp)).exists() else fp
            if not fp.exists():
                continue
            if str(fp) not in cache:
                cache[str(fp)] = project.add_source(fp.name, load_audio(fp), str(fp))
            start, end, tin, tout = (int(ci.findtext(k, "0")) for k in ("start", "end", "in", "out"))
            c = project.add_clip(t.id, cache[str(fp)].id, int(round(start / fps * SR)), int(round(tin / fps * SR)),
                                 int(round((tout - tin) / fps * SR)), ci.findtext("name", fp.stem), push=False)
            m = re.search(r"samples start=(\d+) offset=(\d+) length=(\d+) gain=(-?[\d.]+)", ci.findtext("labels/label2", ""))
            if m:                                           # our own files: restore exact sample positions
                c.start, c.offset, c.length, c.gain_db = int(m[1]), int(m[2]), int(m[3]), float(m[4])
            c.mute = ci.findtext("enabled", "TRUE").upper() == "FALSE"
    project.name = path.stem
    project.changed()


# ============================================================================ AAF
def export_aaf(project, path):
    """AAF composition with one sound slot per track, filler for gaps and the audio embedded in master mobs
    (cuts and positions only; fades, gain and automation are not written)."""
    import aaf2
    path = Path(path)
    tmp = path.with_name(path.stem + "_aaf_tmp")
    media = _consolidate(project, tmp)
    if path.exists():
        path.unlink()
    with aaf2.open(str(path), "w") as f:
        masters = {}
        for sid, wav in media.items():
            m = f.create.MasterMob(project.sources[sid].name)
            f.content.mobs.append(m)
            m.import_audio_essence(str(wav))
            masters[sid] = m
        comp = f.create.CompositionMob(project.name)
        f.content.mobs.append(comp)
        for t in _audio_tracks(project):
            slot = comp.create_sound_slot(edit_rate=SR)
            slot.name = t.name
            seq = f.create.Sequence(media_kind="sound")
            slot.segment = seq
            pos = 0
            for c in sorted(t.clips, key=lambda c: c.start):
                if c.mute or c.length <= 0:
                    continue
                if c.start > pos:
                    seq.components.append(f.create.Filler("sound", int(c.start - pos)))
                elif c.start < pos:
                    continue                                   # overlapping clips cannot share a sequence slot
                seq.components.append(masters[c.source_id].create_source_clip(slot_id=1, start=int(c.offset), length=int(c.length)))
                pos = c.end
    for p in tmp.glob("*"):
        p.unlink()
    tmp.rmdir()
    return path
