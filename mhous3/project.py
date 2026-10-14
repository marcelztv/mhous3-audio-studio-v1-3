"""Multi-track project model and offline mixer (no GUI code - fully testable).

Timeline unit = samples at 48 kHz.  Audio sources are immutable; clips are non-destructive windows onto them."""
import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from . import dsp, fx
from .dsp import SR

PALETTE = ["#5cb6ff", "#4cd4a0", "#ffb454", "#c792ea", "#ff7b8a", "#7fdbda", "#e6db74", "#82aaff"]
FADE_CURVES = ["equal_power", "linear", "equal_gain"]
GRID_CHOICES = {"Off": 0.0, "10 ms": 0.01, "50 ms": 0.05, "100 ms": 0.1, "250 ms": 0.25, "500 ms": 0.5, "1 s": 1.0, "5 s": 5.0}
NUDGE_CHOICES = {"1 ms": 0.001, "10 ms": 0.01, "100 ms": 0.1, "1 s": 1.0}
EDIT_MODES = ["Slip", "Shuffle", "Grid", "Spot"]


@dataclass
class Clip:
    id: int
    source_id: int
    start: int
    offset: int
    length: int
    name: str = ""
    gain_db: float = 0.0
    fade_in: int = 0
    fade_out: int = 0
    fade_in_curve: str = "equal_power"
    fade_out_curve: str = "equal_power"
    mute: bool = False
    lock: bool = False
    loop: bool = False
    pitch: float = 0.0

    @property
    def end(self):
        return self.start + self.length


@dataclass
class Track:
    id: int
    name: str
    color: str = "#5cb6ff"
    gain_db: float = 0.0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    group: int = 0                      # 0 = none
    clips: list = field(default_factory=list)
    auto_volume: list = field(default_factory=list)    # [(sample, dB)] added to the fader
    auto_pan: list = field(default_factory=list)       # [(sample, -1..1)] replaces the pan knob when present
    show_auto: bool = False
    lowcut: float = 0.0
    eq_low: float = 0.0
    eq_mid: float = 0.0
    eq_high: float = 0.0
    gate_thr: float = -80.0             # <= -79 means off
    comp: float = 0.0                   # 0..1
    is_bus: bool = False                # bus / submix track (no clips; receives routed tracks and sends)
    output: int = 0                     # 0 = master, otherwise the id of a bus
    sends: list = field(default_factory=list)     # [{"bus": id, "db": -12.0, "pre": False}]
    inserts: list = field(default_factory=list)   # [{"kind", "params", "bypass", "data"}]
    vca: int = 0                        # 0 = none, otherwise a VCA id
    auto_mode: str = "Read"             # Read / Off / Write / Latch / Touch / Touch/Latch (only Read and Off act offline)
    auto_extra: dict = field(default_factory=dict)  # "mute" -> [(sample, 0|1)], "send:<bus id>" -> [(sample, dB)]


@dataclass
class Source:
    id: int
    name: str
    path: str
    data: np.ndarray


class Project:
    def __init__(self):
        self.sources = {}
        self.tracks = []
        self.name = "Untitled"
        self.path = None
        self._next = 1
        self.undo_stack, self.redo_stack = [], []
        self._listeners = []
        self._last_push = (None, 0.0)
        self.clipboard = []
        self._pitch_cache = {}
        self.vcas = []
        self.master = {"gain_db": 0.0, "mute": False, "inserts": []}
        self.settings = {"out_sr": 48000, "out_bits": 24, "fps": "25", "ruler": "Min:Sec", "tempo": 120.0, "sig_num": 4, "sig_den": 4}

    # ------------------------------------------------------------------ plumbing
    def nid(self):
        self._next += 1
        return self._next - 1

    def add_listener(self, fn):
        self._listeners.append(fn)

    def changed(self):
        for fn in list(self._listeners):
            fn()

    def clear(self):
        self.sources.clear()
        self.tracks.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._pitch_cache.clear()
        self.vcas.clear()
        self.master = {"gain_db": 0.0, "mute": False, "inserts": []}
        self.path, self.name = None, "Untitled"
        self.changed()

    def push_undo(self, key=None):
        """Snapshot for undo.  Calls with the same key within 700 ms are merged (slider drags)."""
        now = time.monotonic()
        if key is not None and self._last_push[0] == key and now - self._last_push[1] < 0.7:
            self._last_push = (key, now)
            return
        self.undo_stack.append(self._snap())
        self.undo_stack = self.undo_stack[-60:]
        self.redo_stack.clear()
        self._last_push = (key, now)

    def _snap(self):
        return copy.deepcopy((self.tracks, self.vcas, self.master))

    def _restore(self, snap):
        tracks, vcas, master = snap
        self.tracks[:] = tracks
        self.vcas[:] = vcas
        self.master = master

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self._snap())
            self._restore(self.undo_stack.pop())
            self.changed()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self._snap())
            self._restore(self.redo_stack.pop())
            self.changed()

    # ------------------------------------------------------------------ lookup
    def track(self, tid):
        return next((t for t in self.tracks if t.id == tid), None)

    def find_clip(self, cid):
        for t in self.tracks:
            for c in t.clips:
                if c.id == cid:
                    return t, c
        return None, None

    def length(self):
        return max((c.end for t in self.tracks for c in t.clips), default=0)

    # ------------------------------------------------------------------ building
    def add_source(self, name, data, path=""):
        s = Source(self.nid(), name, str(path), np.asarray(data, np.float32))
        self.sources[s.id] = s
        return s

    def add_track(self, name=None, push=True):
        if push:
            self.push_undo()
        t = Track(self.nid(), name or f"Track {len(self.tracks) + 1}", PALETTE[len(self.tracks) % len(PALETTE)])
        self.tracks.append(t)
        self.changed()
        return t

    def add_clip(self, track_id, source_id, start=0, offset=0, length=None, name=None, push=True):
        if push:
            self.push_undo()
        src = self.sources[source_id]
        length = len(src.data) - offset if length is None else length
        c = Clip(self.nid(), source_id, int(start), int(offset), int(length), name or src.name)
        self.track(track_id).clips.append(c)
        self.changed()
        return c

    def import_array(self, name, data, path="", track_id=None, start=0):
        """Add audio as a new source and place it on a (new) track."""
        src = self.add_source(name, data, path)
        self.push_undo()
        tr = self.track(track_id) if track_id else self.add_track(Path(name).stem, push=False)
        return self.add_clip(tr.id, src.id, start, push=False)

    def remove_track(self, tid):
        self.push_undo()
        self.tracks[:] = [t for t in self.tracks if t.id != tid]
        self.changed()

    def duplicate_track(self, tid):
        t = self.track(tid)
        if not t:
            return None
        self.push_undo()
        n = copy.deepcopy(t)
        n.id, n.name = self.nid(), t.name + " copy"
        for c in n.clips:
            c.id = self.nid()
        self.tracks.insert(self.tracks.index(t) + 1, n)
        self.changed()
        return n

    # ------------------------------------------------------------------ grid / snapping
    def snap(self, sample, grid_s):
        if grid_s <= 0:
            return int(sample)
        g = grid_s * SR
        return int(round(sample / g) * g)

    # ------------------------------------------------------------------ clip audio
    def _segment(self, clip):
        src = self.sources[clip.source_id].data
        n = clip.length
        if clip.loop and clip.offset + n > len(src):
            base = src[clip.offset:]
            if len(base) == 0:
                return np.zeros((n, src.shape[1]), np.float32)
            reps = int(np.ceil(n / len(base)))
            seg = np.tile(base, (reps, 1))[:n]
        else:
            seg = src[clip.offset:clip.offset + n]
            if len(seg) < n:
                seg = np.pad(seg, ((0, n - len(seg)), (0, 0)))
        if abs(clip.pitch) > 0.01:
            key = (clip.source_id, clip.offset, clip.length, round(clip.pitch, 2), clip.loop)
            if key not in self._pitch_cache:
                self._pitch_cache[key] = dsp.pitch_shift(seg, clip.pitch)
                if len(self._pitch_cache) > 12:
                    self._pitch_cache.pop(next(iter(self._pitch_cache)))
            seg = self._pitch_cache[key]
        if seg.shape[1] == 1:
            seg = np.repeat(seg, 2, axis=1)
        return seg[:, :2]

    def effective_fades(self, track):
        """Fade lengths per clip including automatic crossfades where clips on the same track overlap."""
        eff = {c.id: [c.fade_in, c.fade_out] for c in track.clips}
        cl = sorted(track.clips, key=lambda c: c.start)
        for a, b in zip(cl, cl[1:]):
            ov = min(a.end, b.end) - b.start
            if ov > 0:
                eff[a.id][1] = max(eff[a.id][1], ov)
                eff[b.id][0] = max(eff[b.id][0], ov)
        return eff

    def clip_audio(self, clip, fades=None):
        seg = self._segment(clip).copy()
        if clip.mute:
            return seg * 0
        seg *= float(dsp.db2lin(clip.gain_db))
        fi, fo = fades if fades else (clip.fade_in, clip.fade_out)
        n = len(seg)
        if fi > 0:
            fi = min(fi, n)
            seg[:fi] *= dsp.fade_curve(fi, clip.fade_in_curve, True)[:, None]
        if fo > 0:
            fo = min(fo, n)
            seg[n - fo:] *= dsp.fade_curve(fo, clip.fade_out_curve, False)[:, None]
        return seg

    # ------------------------------------------------------------------ track / mix rendering
    @staticmethod
    def _curve(points, n, default, db=False):
        pts = sorted(points)
        xs = np.array([p[0] for p in pts], np.float64)
        ys = np.array([p[1] for p in pts], np.float64)
        v = np.interp(np.arange(n), xs, ys)
        return v

    def _vca(self, track):
        v = next((x for x in self.vcas if x["id"] == track.vca), None) if track.vca else None
        return (v["gain_db"], v["mute"]) if v else (0.0, False)

    def _track_signal(self, track, n, source=None):
        """Returns (pre_fader, post_fader) for a track.  `source` is the summed input of a bus; clips are used otherwise."""
        if source is not None:
            buf = source.copy()
        else:
            buf = np.zeros((n, 2), np.float32)
            eff = self.effective_fades(track)
            for c in track.clips:
                if c.start >= n or c.length <= 0:
                    continue
                seg = self.clip_audio(c, eff[c.id])
                s0 = max(0, c.start)
                a = s0 - c.start
                e = min(n, c.end)
                buf[s0:e] += seg[a:a + (e - s0)]
        if track.lowcut > 0:
            buf = dsp.highpass(buf, track.lowcut)
        if track.eq_low or track.eq_mid or track.eq_high:
            buf = dsp.parametric_eq(buf, [("lowshelf", 150, track.eq_low, 0.7), ("peak", 1500, track.eq_mid, 0.8),
                                          ("highshelf", 6000, track.eq_high, 0.7)])
        if track.gate_thr > -79:
            buf = dsp.expander(buf, SR, track.gate_thr, 2.5, -24.0, 5.0, 120.0)
        if track.comp > 0.01:
            buf = dsp.compressor(buf, SR, -24, 1 + 3 * track.comp, 10, 120, 6, 0.0)
        if track.inserts:
            buf = fx.process_chain(buf, track.inserts)
        pre = buf.astype(np.float32)
        use_auto = track.auto_mode != "Off"
        vca_db, vca_mute = self._vca(track)
        base_db = track.gain_db + vca_db
        if use_auto and track.auto_volume:
            g = dsp.db2lin(self._curve(track.auto_volume, n, 0.0) + base_db).astype(np.float32)
            out = pre * g[:, None]
        else:
            out = pre * float(dsp.db2lin(base_db))
        if use_auto and track.auto_pan:
            pan = self._curve(track.auto_pan, n, 0.0).astype(np.float32)
            out = np.stack([out[:, 0] * np.minimum(1.0, 1.0 - pan), out[:, 1] * np.minimum(1.0, 1.0 + pan)], axis=1)
        elif abs(track.pan) > 1e-4:
            out = out * np.array([min(1.0, 1.0 - track.pan), min(1.0, 1.0 + track.pan)], np.float32)
        if use_auto and track.auto_extra.get("mute"):
            pts = sorted(track.auto_extra["mute"])
            xs = np.array([p[0] for p in pts])
            vals = np.array([p[1] for p in pts], np.float32)
            idx = np.searchsorted(xs, np.arange(n), side="right") - 1
            state = np.where(idx >= 0, vals[np.clip(idx, 0, None)], 0.0)
            smooth = np.convolve(1.0 - state, np.ones(240) / 240, mode="same").astype(np.float32)
            out = out * smooth[:, None]
        if vca_mute:
            out = out * 0
        return pre, out.astype(np.float32)

    def render_track(self, track, length=None, apply_mute=True):
        n = self.length() if length is None else length
        _, post = self._track_signal(track, n)
        return post * 0 if (apply_mute and track.mute) else post

    def audible(self, track):
        if track.is_bus:
            return not track.mute
        if any(t.solo for t in self.tracks if not t.is_bus):
            return track.solo
        return not track.mute

    def bus_order(self):
        """Buses ordered so that a bus is processed before the bus it feeds (cycles fall back to the master)."""
        buses = {t.id: t for t in self.tracks if t.is_bus}

        def depth(t, seen):
            if t.output in buses and t.output not in seen:
                return 1 + depth(buses[t.output], seen | {t.output})
            return 0
        return sorted(buses.values(), key=lambda t: -depth(t, {t.id}))

    def _route_target(self, t):
        buses = {x.id for x in self.tracks if x.is_bus}
        if t.output in buses and t.output != t.id:
            tgt = self.track(t.output)
            seen, cur = {t.id}, tgt
            while cur is not None and cur.output in buses:      # cycle check
                if cur.output in seen:
                    return 0
                seen.add(cur.id)
                cur = self.track(cur.output)
            return t.output
        return 0

    def _send_curve(self, t, s, n):
        pts = t.auto_extra.get(f"send:{s['bus']}")
        if pts and t.auto_mode != "Off":
            return dsp.db2lin(self._curve(pts, n, s["db"])).astype(np.float32)
        return float(dsp.db2lin(s["db"]))

    def mixdown(self, include=None):
        """Master output.  Tracks -> (sends) -> buses -> master, then master inserts and fader."""
        n = self.length()
        master = np.zeros((n, 2), np.float32)
        if n == 0:
            return master
        bus_in = {t.id: np.zeros((n, 2), np.float32) for t in self.tracks if t.is_bus}
        for t in self.tracks:
            if t.is_bus or not self.audible(t):
                continue
            pre, post = self._track_signal(t, n)
            for s in t.sends:
                if s["bus"] in bus_in:
                    tap = pre if s.get("pre") else post
                    g = self._send_curve(t, s, n)
                    bus_in[s["bus"]] += tap * (g[:, None] if isinstance(g, np.ndarray) else g)
            tgt = self._route_target(t)
            (bus_in[tgt] if tgt else master).__iadd__(post)
        for b in self.bus_order():
            _, post = self._track_signal(b, n, source=bus_in[b.id])
            if self.audible(b):
                tgt = self._route_target(b)
                (bus_in[tgt] if tgt else master).__iadd__(post)
        if self.master["inserts"]:
            master = fx.process_chain(master, self.master["inserts"])
        master = master * float(dsp.db2lin(self.master["gain_db"]))
        return master * 0 if self.master["mute"] else master

    def render_stems(self, folder, subtype="PCM_24", with_buses=True):
        """One WAV per track (and bus) - post fader, inserts, automation, pan - all the same length so they line up."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        n = self.length()
        paths = []
        if n == 0:
            return paths
        bus_in = {t.id: np.zeros((n, 2), np.float32) for t in self.tracks if t.is_bus}
        outs = {}
        for t in self.tracks:
            if t.is_bus:
                continue
            pre, post = self._track_signal(t, n)
            outs[t.id] = post
            for s in t.sends:
                if s["bus"] in bus_in:
                    bus_in[s["bus"]] += (pre if s.get("pre") else post) * self._send_curve(t, s, n)
            tgt = self._route_target(t)
            if tgt:
                bus_in[tgt] += post
        for b in self.bus_order():
            _, post = self._track_signal(b, n, source=bus_in[b.id])
            outs[b.id] = post
            tgt = self._route_target(b)
            if tgt:
                bus_in[tgt] += post
        for i, t in enumerate(self.tracks, 1):
            if t.is_bus and not with_buses:
                continue
            safe = "".join(ch if ch.isalnum() or ch in " _-" else "_" for ch in t.name).strip() or f"track{i}"
            p = folder / f"{i:02d}_{'BUS_' if t.is_bus else ''}{safe}.wav"
            sf.write(p, outs[t.id], SR, subtype=subtype)
            paths.append(p)
        return paths

    # ------------------------------------------------------------------ mixer: buses, sends, VCAs, inserts
    def add_bus(self, name=None):
        self.push_undo()
        t = Track(self.nid(), name or f"Bus {sum(1 for x in self.tracks if x.is_bus) + 1}", "#c792ea", is_bus=True)
        self.tracks.append(t)
        self.changed()
        return t

    def add_vca(self, name=None):
        self.push_undo()
        v = {"id": self.nid(), "name": name or f"VCA {len(self.vcas) + 1}", "gain_db": 0.0, "mute": False}
        self.vcas.append(v)
        self.changed()
        return v

    def set_route(self, tid, bus_id):
        self.set_track(tid, output=int(bus_id))

    def set_send(self, tid, bus_id, db=-12.0, pre=False):
        t = self.track(tid)
        self.push_undo("send")
        for s in t.sends:
            if s["bus"] == bus_id:
                s["db"], s["pre"] = float(db), bool(pre)
                break
        else:
            t.sends.append({"bus": int(bus_id), "db": float(db), "pre": bool(pre)})
        self.changed()

    def remove_send(self, tid, bus_id):
        t = self.track(tid)
        self.push_undo()
        t.sends[:] = [s for s in t.sends if s["bus"] != bus_id]
        self.changed()

    def _chain(self, tid):
        return self.master["inserts"] if tid == 0 else self.track(tid).inserts

    def add_insert(self, tid, kind, params=None, data=None):
        self.push_undo()
        slot = {"kind": kind, "params": dict(params if params is not None else fx.defaults(kind) if kind in fx.EFFECTS else {}),
                "bypass": False, "data": dict(data or {})}
        self._chain(tid).append(slot)
        self.changed()
        return slot

    def remove_insert(self, tid, index):
        ch = self._chain(tid)
        if 0 <= index < len(ch):
            self.push_undo()
            del ch[index]
            self.changed()

    def move_insert(self, tid, index, delta):
        ch = self._chain(tid)
        j = index + delta
        if 0 <= index < len(ch) and 0 <= j < len(ch):
            self.push_undo()
            ch[index], ch[j] = ch[j], ch[index]
            self.changed()

    def set_insert(self, tid, index, **kw):
        ch = self._chain(tid)
        if 0 <= index < len(ch):
            self.push_undo("insert")
            for k, v in kw.items():
                if k == "params":
                    ch[index]["params"].update(v)
                else:
                    ch[index][k] = v
            self.changed()

    def load_track_preset(self, tid, name):
        self.push_undo()
        self._chain(tid)[:] = copy.deepcopy(fx.TRACK_PRESETS[name])
        self.changed()

    def new_from_template(self, name):
        self.clear()
        for tname, preset in fx.SESSION_TEMPLATES[name]:
            t = self.add_track(tname, push=False)
            if preset:
                t.inserts[:] = copy.deepcopy(fx.TRACK_PRESETS[preset])
        self.name = name
        self.changed()

    # ------------------------------------------------------------------ edit operations
    def _unlocked(self, ids):
        out = []
        for cid in ids:
            t, c = self.find_clip(cid)
            if c and not c.lock:
                out.append((t, c))
        return out

    def split(self, ids, at):
        """Split every unlocked clip in `ids` that spans sample `at`.  Returns the new right-hand clip ids."""
        new = []
        todo = [(t, c) for t, c in self._unlocked(ids) if c.start < at < c.end]
        if not todo:
            return new
        self.push_undo()
        for t, c in todo:
            cut = at - c.start
            r = copy.deepcopy(c)
            r.id, r.start, r.offset, r.length = self.nid(), at, c.offset + cut, c.length - cut
            r.fade_in = 0
            r.pitch = c.pitch
            c.length = cut
            c.fade_out = 0
            t.clips.append(r)
            new.append(r.id)
        self.changed()
        return new

    def split_range(self, ids, a, b):
        self.split(ids, a)
        self.split(ids + [c.id for t in self.tracks for c in t.clips], b)

    def separate_at_grid(self, ids, grid_s):
        if grid_s <= 0:
            return
        g = int(grid_s * SR)
        self.push_undo()
        for cid in list(ids):
            t, c = self.find_clip(cid)
            if not c or c.lock:
                continue
            pos = (c.start // g + 1) * g
            cur = c
            while pos < cur.end:
                nid = self._split_one(t, cur, pos)
                cur = nid
                pos += g
        self.changed()

    def _split_one(self, t, c, at):
        cut = at - c.start
        r = copy.deepcopy(c)
        r.id, r.start, r.offset, r.length = self.nid(), at, c.offset + cut, c.length - cut
        r.fade_in = 0
        c.length, c.fade_out = cut, 0
        t.clips.append(r)
        return r

    def separate_at_transients(self, ids):
        self.push_undo()
        count = 0
        for cid in list(ids):
            t, c = self.find_clip(cid)
            if not c or c.lock:
                continue
            seg = self._segment(c).mean(axis=1)
            cur = c
            for p in detect_pts(seg):
                at = c.start + p
                if cur.start + 400 < at < cur.end - 400:
                    cur = self._split_one(t, cur, max(cur.start + 1, at - int(0.01 * SR)))
                    count += 1
        self.changed()
        return count

    def delete(self, ids, shuffle=False):
        todo = self._unlocked(ids)
        if not todo:
            return
        self.push_undo()
        tracks = set()
        for t, c in todo:
            t.clips.remove(c)
            tracks.add(t.id)
        if shuffle:
            for tid in tracks:
                self._repack(self.track(tid))
        self.changed()

    def _repack(self, t):
        pos = None
        for c in sorted(t.clips, key=lambda c: c.start):
            if pos is None:
                pos = c.start
            c.start = pos
            pos = c.end

    def copy_clips(self, ids):
        sel = [(t, c) for t in self.tracks for c in t.clips if c.id in ids]
        if not sel:
            return
        t0 = min(c.start for _, c in sel)
        ti = {t.id: i for i, t in enumerate(self.tracks)}
        r0 = min(ti[t.id] for t, _ in sel)
        self.clipboard = [(ti[t.id] - r0, copy.deepcopy(c), c.start - t0) for t, c in sel]

    def paste(self, track_index, at):
        if not self.clipboard:
            return []
        self.push_undo()
        new = []
        for dr, c, rel in self.clipboard:
            idx = min(max(0, track_index + dr), len(self.tracks) - 1)
            n = copy.deepcopy(c)
            n.id, n.start, n.lock = self.nid(), int(at + rel), False
            self.tracks[idx].clips.append(n)
            new.append(n.id)
        self.changed()
        return new

    def duplicate(self, ids):
        self.copy_clips(ids)
        if not self.clipboard:
            return []
        sel = [c for t in self.tracks for c in t.clips if c.id in ids]
        end = max(c.end for c in sel)
        t0 = min(c.start for c in sel)
        ti = self.tracks.index(self.find_clip(sel[0].id)[0])
        return self.paste(ti, end) if end > t0 else []

    def repeat(self, ids, times):
        out = []
        for _ in range(max(1, int(times))):
            out = self.duplicate(out or ids)
        return out

    def move(self, cid, new_start, new_track_id=None, shuffle=False):
        t, c = self.find_clip(cid)
        if not c or c.lock:
            return
        c.start = max(0, int(new_start))
        if new_track_id and new_track_id != t.id:
            nt = self.track(new_track_id)
            if nt:
                t.clips.remove(c)
                nt.clips.append(c)
                t = nt
        if shuffle:
            self._repack(t)

    def nudge(self, ids, delta):
        todo = self._unlocked(ids)
        if todo:
            self.push_undo("nudge")
            for _, c in todo:
                c.start = max(0, c.start + int(delta))
            self.changed()

    def trim_start(self, cid, new_start):
        t, c = self.find_clip(cid)
        if not c or c.lock:
            return
        new_start = int(min(max(new_start, 0), c.end - 64))
        d = new_start - c.start
        if not c.loop and c.offset + d < 0:
            d = -c.offset
        c.start += d
        c.offset = max(0, c.offset + d)
        c.length -= d
        c.fade_in = min(c.fade_in, c.length)

    def trim_end(self, cid, new_end, allow_loop=False):
        t, c = self.find_clip(cid)
        if not c or c.lock:
            return
        new_len = int(max(64, new_end - c.start))
        src_len = len(self.sources[c.source_id].data) - c.offset
        if new_len > src_len and not c.loop:
            if allow_loop:
                c.loop = True
            else:
                new_len = src_len
        c.length = new_len
        c.fade_out = min(c.fade_out, c.length)

    def trim_to(self, ids, a, b):
        """Standard trim to a time range: keep only [a, b) of each selected clip."""
        todo = self._unlocked(ids)
        if todo:
            self.push_undo()
            for t, c in todo:
                if b is not None and b < c.end:
                    self.trim_end(c.id, b)
                if a is not None and a > c.start:
                    self.trim_start(c.id, a)
            self.changed()

    def set_fades(self, ids, fade_in=None, fade_out=None, in_curve=None, out_curve=None, push=True):
        todo = self._unlocked(ids)
        if not todo:
            return
        if push:
            self.push_undo("fades")
        for _, c in todo:
            if fade_in is not None:
                c.fade_in = int(min(max(0, fade_in), c.length))
            if fade_out is not None:
                c.fade_out = int(min(max(0, fade_out), c.length))
            if in_curve:
                c.fade_in_curve = in_curve
            if out_curve:
                c.fade_out_curve = out_curve
        self.changed()

    def set_clip(self, ids, **kw):
        todo = self._unlocked(ids) if not set(kw) <= {"lock"} else [(None, self.find_clip(i)[1]) for i in ids]
        if todo:
            self.push_undo("clip:" + ",".join(kw))
            for _, c in todo:
                if c:
                    for k, v in kw.items():
                        setattr(c, k, v)
            self.changed()

    def set_track(self, tid, **kw):
        t = self.track(tid)
        if t:
            self.push_undo("track:" + ",".join(kw))
            for k, v in kw.items():
                setattr(t, k, v)
            self.changed()

    # ------------------------------------------------------------------ automation
    def auto_add(self, tid, kind, sample, value):
        t = self.track(tid)
        pts = t.auto_volume if kind == "volume" else t.auto_pan
        self.push_undo()
        pts.append((int(sample), float(value)))
        pts.sort()
        self.changed()

    def auto_move(self, tid, kind, index, sample, value):
        t = self.track(tid)
        pts = t.auto_volume if kind == "volume" else t.auto_pan
        if 0 <= index < len(pts):
            pts[index] = (int(max(0, sample)), float(value))
            pts.sort()

    def auto_remove(self, tid, kind, index):
        t = self.track(tid)
        pts = t.auto_volume if kind == "volume" else t.auto_pan
        if 0 <= index < len(pts):
            self.push_undo()
            del pts[index]
            self.changed()

    # ------------------------------------------------------------------ save / load with media consolidation
    def save(self, path):
        """Writes <name>.mh3p (JSON) and copies every source into <name>_media/ so renaming or moving the original
        files can never break the project (background-safe 'consolidation')."""
        path = Path(path)
        media = path.with_name(path.stem + "_media")
        media.mkdir(parents=True, exist_ok=True)
        srcs = {}
        for s in self.sources.values():
            f = media / f"source_{s.id}.wav"
            if not f.exists():
                sf.write(f, s.data, SR, subtype="FLOAT")
            srcs[str(s.id)] = {"name": s.name, "orig": s.path, "file": f"{media.name}/{f.name}"}
        doc = {"format": "mh3p", "version": 2, "name": path.stem, "next": self._next, "sources": srcs,
               "tracks": [asdict(t) for t in self.tracks], "vcas": self.vcas, "master": self.master, "settings": self.settings}
        path.write_text(json.dumps(doc, indent=1), "utf-8")
        self.path, self.name = str(path), path.stem
        return path

    def load(self, path):
        path = Path(path)
        doc = json.loads(path.read_text("utf-8"))
        self.clear()
        for sid, info in doc["sources"].items():
            data, sr = sf.read(path.parent / info["file"], dtype="float32", always_2d=True)
            s = Source(int(sid), info["name"], info.get("orig", ""), data)
            self.sources[s.id] = s
        for td in doc["tracks"]:
            clips = [Clip(**c) for c in td.pop("clips")]
            td["auto_volume"] = [tuple(p) for p in td.get("auto_volume", [])]
            td["auto_pan"] = [tuple(p) for p in td.get("auto_pan", [])]
            td["auto_extra"] = {k: [tuple(p) for p in v] for k, v in td.get("auto_extra", {}).items()}
            t = Track(**{**td, "clips": clips})
            self.tracks.append(t)
        self._next = doc.get("next", 1000)
        self.vcas[:] = doc.get("vcas", [])
        self.master = doc.get("master", {"gain_db": 0.0, "mute": False, "inserts": []})
        self.settings.update(doc.get("settings", {}))
        self.path, self.name = str(path), doc.get("name", path.stem)
        self.changed()


def detect_pts(mono):
    return dsp.detect_transients(mono)
