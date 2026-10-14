"""Multi-track timeline (Precision / Full Studio): tracks, clips, trims, fades, clip gain, automation lanes."""
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QMenu, QMessageBox,
                               QPushButton, QScrollBar, QSlider, QToolButton, QVBoxLayout, QWidget, QFormLayout, QGroupBox)

from . import dsp, engine, theme
from .dsp import SR
from . import interchange as ix
from .project import EDIT_MODES, FADE_CURVES, GRID_CHOICES, NUDGE_CHOICES, PALETTE
from .session import Task
from .widgets import tip

HEADER_W, RULER_H, TRACK_H, AUTO_H = 214, 28, 96, 64
DB_MIN, DB_MAX = -60.0, 12.0
MIP = 256


def _mip(mono):
    n = len(mono) // MIP * MIP
    if n == 0:
        return None
    b = mono[:n].reshape(-1, MIP)
    return b.min(axis=1), b.max(axis=1)


class TimelineView(QWidget):
    selectionChanged = Signal()
    seekRequested = Signal(int)
    filesDropped = Signal(list, int, int)
    spotRequested = Signal(int)
    contextRequested = Signal(object)

    def __init__(self, project, session):
        super().__init__()
        self.p, self.session = project, session
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)
        self.pps = 70.0              # pixels per second
        self.t0 = 0.0                # seconds at the left edge of the lanes
        self.y0 = 0
        self.sel_clips, self.sel_track, self.range = set(), None, None
        self.mode, self.grid = "Slip", 0.0
        self.drag = None
        self._mono, self._mips = {}, {}
        self._last_ph = -1
        project.add_listener(self.update)
        session.tick.connect(self._tick)

    # ------------------------------------------------------------------ geometry
    def total_seconds(self):
        return max(10.0, self.p.length() / SR + 5)

    def sx(self, sample):
        return HEADER_W + (sample / SR - self.t0) * self.pps

    def tx(self, x):
        return int((self.t0 + (x - HEADER_W) / self.pps) * SR)

    def track_h(self, t):
        return TRACK_H + (AUTO_H if t.show_auto else 0)

    def tops(self):
        y, out = RULER_H - self.y0, []
        for t in self.p.tracks:
            out.append(y)
            y += self.track_h(t)
        return out, y

    def track_at(self, y):
        tops, _ = self.tops()
        for i, t in enumerate(self.p.tracks):
            if tops[i] <= y < tops[i] + self.track_h(t):
                return i, y - tops[i]
        return -1, 0

    def clip_rect(self, t, c, top):
        return QRectF(self.sx(c.start), top + 4, max(2.0, c.length / SR * self.pps), TRACK_H - 8)

    def snap(self, sample):
        return self.p.snap(sample, self.grid) if self.mode == "Grid" or self.grid > 0 else int(sample)

    def set_t0(self, t):
        self.t0 = max(0.0, t)
        self.update()
        self.selectionChanged.emit()

    def zoom(self, factor, cx=None):
        cx = HEADER_W + (self.width() - HEADER_W) / 2 if cx is None else cx
        anchor = self.t0 + (cx - HEADER_W) / self.pps
        self.pps = min(4000.0, max(2.0, self.pps * factor))
        self.t0 = max(0.0, anchor - (cx - HEADER_W) / self.pps)
        self.update()
        self.selectionChanged.emit()

    def zoom_fit(self):
        self.pps = max(2.0, (self.width() - HEADER_W - 30) / max(self.p.length() / SR, 5))
        self.t0 = 0.0
        self.update()
        self.selectionChanged.emit()

    def _tick(self):
        ph = self.session.global_pos()
        if ph != self._last_ph:
            self._last_ph = ph
            if self.session.player.playing:
                x = self.sx(ph)
                if x > self.width() - 40 or x < HEADER_W:
                    self.t0 = max(0.0, ph / SR - 1.0)
                    self.selectionChanged.emit()
            self.update()

    # ------------------------------------------------------------------ peaks
    def _peaks(self, sid):
        if sid not in self._mono:
            d = self.p.sources[sid].data
            self._mono[sid] = d.mean(axis=1).astype(np.float32)
            self._mips[sid] = _mip(self._mono[sid])
        return self._mono[sid], self._mips[sid]

    def _wave_cols(self, c, xa, xb):
        mono, mip = self._peaks(c.source_id)
        w = max(1, int(xb - xa))
        spp = SR / self.pps
        s0 = (xa - self.sx(c.start)) / self.pps * SR + c.offset
        edges = s0 + np.arange(w + 1) * spp
        lo_lim, hi_lim = c.offset, c.offset + c.length
        if c.loop:
            edges = c.offset + ((edges - c.offset) % max(1, len(mono) - c.offset))
        edges = np.clip(edges, lo_lim, min(hi_lim, len(mono)))
        if spp >= 2 * MIP and mip is not None:
            e = (edges / MIP).astype(np.int64)
            arr_lo, arr_hi, n = mip[0], mip[1], len(mip[0])
        else:
            e = edges.astype(np.int64)
            arr_lo = arr_hi = mono
            n = len(mono)
        e = np.clip(e, 0, n)
        a, b = int(e[0]), int(max(e[-1], e[0] + 1))
        if a >= n:
            return np.zeros(w), np.zeros(w)
        idx = np.minimum(e[:-1] - a, max(0, min(b, n) - a - 1))
        lo = np.minimum.reduceat(arr_lo[a:min(b, n)], idx)
        hi = np.maximum.reduceat(arr_hi[a:min(b, n)], idx)
        return lo, hi

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pal = theme.palette()
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, theme.color("panel2"))
        tops, _ = self.tops()
        font = QFont(self.font())
        font.setPixelSize(12)
        p.setFont(font)
        if not self.p.tracks:
            p.setPen(theme.color("muted"))
            p.drawText(self.rect(), Qt.AlignCenter, "Drop audio files here, or click Import / Add Track.\nOpen a file first and it appears as Track 1.")
        for i, t in enumerate(self.p.tracks):
            top, h = tops[i], self.track_h(t)
            if top + h < RULER_H or top > H:
                continue
            p.fillRect(HEADER_W, int(top), W - HEADER_W, h, theme.color("panel" if i % 2 == 0 else "panel2"))
            p.setPen(QPen(theme.color("border"), 1))
            p.drawLine(0, int(top + h), W, int(top + h))
            self._paint_clips(p, t, top)
            if t.show_auto:
                self._paint_auto(p, t, top + TRACK_H)
        self._paint_grid(p, H)
        for i, t in enumerate(self.p.tracks):
            if tops[i] + self.track_h(t) >= RULER_H and tops[i] <= H:
                self._paint_header(p, t, i, tops[i])
        # range + playhead
        if self.range:
            a, b = self.sx(self.range[0]), self.sx(self.range[1])
            p.fillRect(QRectF(a, RULER_H, max(1, b - a), H - RULER_H), theme.color("accent", 45))
        self._paint_ruler(p, W)
        x = self.sx(self.session.global_pos())
        if HEADER_W <= x <= W:
            p.setPen(QPen(theme.color("text"), 2))
            p.drawLine(int(x), 0, int(x), H)
            tri = QPainterPath()
            tri.moveTo(x - 6, 0)
            tri.lineTo(x + 6, 0)
            tri.lineTo(x, 10)
            p.fillPath(tri, theme.color("accent"))
        if self.drag and self.drag.get("kind") == "band":
            r = QRectF(self.drag["a"], self.drag["b"]).normalized()
            p.setPen(QPen(theme.color("accent"), 1, Qt.DashLine))
            p.fillRect(r, theme.color("accent", 40))
            p.drawRect(r)

    def _paint_grid(self, p, H):
        if self.grid <= 0 and self.pps < 40:
            return
        step = self.grid if self.grid > 0 else (1.0 if self.pps < 200 else 0.1)
        if step * self.pps < 6:
            return
        t = math.floor(self.t0 / step) * step
        p.setPen(QPen(theme.color("grid"), 1))
        while self.sx(t * SR) < self.width():
            x = self.sx(t * SR)
            if x >= HEADER_W:
                p.drawLine(int(x), RULER_H, int(x), H)
            t += step

    def _ruler_label(self, t, step):
        st = self.p.settings
        mode = st.get("ruler", "Min:Sec")
        if mode == "Timecode":
            return ix.frames_to_tc(ix.sec_to_frames(max(t, 0), st.get("fps", "25")), st.get("fps", "25"))
        if mode == "Bars|Beats":
            beat = 60.0 / max(st.get("tempo", 120.0), 1)
            tb = t / beat
            num = int(st.get("sig_num", 4))
            return f"{int(tb // num) + 1}|{int(tb % num) + 1}|{int(round((tb % 1) * 960)) % 960:03d}"
        m, s = divmod(t, 60)
        return f"{int(m)}:{s:05.2f}" if step < 1 else f"{int(m)}:{int(s):02d}"

    def _paint_ruler(self, p, W):
        p.fillRect(0, 0, W, RULER_H, theme.color("panel"))
        p.setPen(QPen(theme.color("border"), 1))
        p.drawLine(0, RULER_H, W, RULER_H)
        st = self.p.settings
        if st.get("ruler") == "Bars|Beats":
            beat = 60.0 / max(st.get("tempo", 120.0), 1)
            num = int(st.get("sig_num", 4))
            steps = [beat * k for k in (0.25, 0.5, 1, num, 2 * num, 4 * num, 8 * num, 16 * num, 32 * num)]
        else:
            steps = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600)
        step = next((s for s in steps if s * self.pps >= 76), steps[-1] * 2)
        t = math.floor(self.t0 / step) * step
        p.setPen(theme.color("muted"))
        while self.sx(t * SR) < W:
            x = self.sx(t * SR)
            if x >= HEADER_W:
                p.drawLine(int(x), RULER_H - 8, int(x), RULER_H)
                p.drawText(QPointF(x + 3, RULER_H - 10), self._ruler_label(t, step))
            t += step
        p.fillRect(0, 0, HEADER_W, RULER_H, theme.color("panel"))
        p.setPen(theme.color("muted"))
        p.drawText(QRectF(10, 0, HEADER_W - 10, RULER_H), Qt.AlignVCenter, f"Tracks: {len(self.p.tracks)}   Grid: {'off' if self.grid <= 0 else self.grid}")

    def _paint_header(self, p, t, i, top):
        h = self.track_h(t)
        p.fillRect(QRectF(0, top, HEADER_W, h), theme.color("panel"))
        sel = self.sel_track == t.id
        if sel:
            p.fillRect(QRectF(0, top, HEADER_W, h), theme.color("accent", 28))
        p.fillRect(QRectF(0, top, 6, h), QColor(t.color))
        p.setPen(QPen(theme.color("border"), 1))
        p.drawLine(HEADER_W - 1, int(top), HEADER_W - 1, int(top + h))
        f = QFont(self.font())
        f.setPixelSize(13)
        f.setBold(True)
        p.setFont(f)
        p.setPen(theme.color("text"))
        grp = f"   [G{t.group}]" if t.group else ""
        p.drawText(QRectF(14, top + 6, HEADER_W - 20, 20), Qt.AlignVCenter, t.name + grp)
        f.setBold(False)
        f.setPixelSize(11)
        p.setFont(f)
        for k, (label, on, color) in enumerate((("M", t.mute, "warn"), ("S", t.solo, "good"), ("A", t.show_auto, "accent"))):
            r = QRectF(14 + k * 34, top + 30, 28, 20)
            p.setPen(QPen(theme.color("border"), 1))
            p.setBrush(theme.color(color) if on else theme.color("panel2"))
            p.drawRoundedRect(r, 5, 5)
            p.setPen(theme.color("on_accent") if on else theme.color("text"))
            p.drawText(r, Qt.AlignCenter, label)
        # fader and pan bars
        for row, (label, frac, val) in enumerate((("Vol", (t.gain_db - DB_MIN) / (DB_MAX - DB_MIN), f"{t.gain_db:+.1f} dB"),
                                                  ("Pan", (t.pan + 1) / 2, "C" if abs(t.pan) < 0.02 else (f"L{int(-t.pan * 100)}" if t.pan < 0 else f"R{int(t.pan * 100)}")))):
            y = top + 60 + row * 18 if False else top + 56 + row * 18
            p.setPen(theme.color("muted"))
            p.drawText(QRectF(14, y - 2, 26, 14), Qt.AlignVCenter, label)
            bar = QRectF(42, y + 3, HEADER_W - 42 - 74, 5)
            p.setPen(Qt.NoPen)
            p.setBrush(theme.color("border"))
            p.drawRoundedRect(bar, 2.5, 2.5)
            p.setBrush(theme.color("accent"))
            p.drawEllipse(QPointF(bar.left() + bar.width() * min(1, max(0, frac)), bar.center().y()), 6, 6)
            p.setPen(theme.color("muted"))
            p.drawText(QRectF(bar.right() + 8, y - 2, 58, 14), Qt.AlignVCenter | Qt.AlignRight, val)

    def _paint_clips(self, p, t, top):
        eff = self.p.effective_fades(t)
        W = self.width()
        for c in t.clips:
            r = self.clip_rect(t, c, top)
            if r.right() < HEADER_W or r.left() > W:
                continue
            col = QColor(t.color)
            sel = c.id in self.sel_clips
            fill = QColor(col.red(), col.green(), col.blue(), 120 if sel else 70)
            if c.mute:
                fill = QColor(120, 120, 120, 50)
            clip_path = QPainterPath()
            clip_path.addRoundedRect(r, 6, 6)
            p.save()
            p.setClipRect(QRectF(HEADER_W, 0, W - HEADER_W, self.height()))
            p.fillPath(clip_path, fill)
            xa, xb = max(r.left(), HEADER_W), min(r.right(), W)
            if xb - xa >= 1:
                lo, hi = self._wave_cols(c, int(xa), int(xb))
                mid, amp = r.center().y() + 6, (r.height() - 22) / 2
                gscale = float(dsp.db2lin(c.gain_db))
                pen = QPen(col.lighter(130) if not c.mute else QColor(150, 150, 150), 1)
                p.setPen(pen)
                for k in range(len(lo)):
                    y1 = mid - float(np.clip(hi[k] * gscale, -1, 1)) * amp
                    y2 = mid - float(np.clip(lo[k] * gscale, -1, 1)) * amp
                    p.drawLine(int(xa) + k, int(y1), int(xa) + k, int(max(y2, y1 + 1)))
            # fades
            fi, fo = eff[c.id]
            p.setPen(QPen(theme.color("text", 200), 1.4))
            if fi > 0:
                w = fi / SR * self.pps
                path = QPainterPath()
                path.moveTo(r.left(), r.bottom() - 2)
                for k in range(0, 21):
                    u = k / 20
                    g = math.sin(u * math.pi / 2) if c.fade_in_curve == "equal_power" else u
                    path.lineTo(r.left() + w * u, r.bottom() - 2 - g * (r.height() - 4))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
            if fo > 0:
                w = fo / SR * self.pps
                path = QPainterPath()
                path.moveTo(r.right() - w, r.top() + 2)
                for k in range(0, 21):
                    u = k / 20
                    g = math.cos(u * math.pi / 2) if c.fade_out_curve == "equal_power" else 1 - u
                    path.lineTo(r.right() - w + w * u, r.bottom() - 2 - g * (r.height() - 4))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
            # gain line
            gy = r.top() + (DB_GAIN_MAX - c.gain_db) / (DB_GAIN_MAX - DB_GAIN_MIN) * r.height()
            p.setPen(QPen(theme.color("warn", 190), 1, Qt.DashLine))
            p.drawLine(QPointF(r.left(), gy), QPointF(r.right(), gy))
            # fade handles + label
            p.setPen(QPen(theme.color("text"), 1))
            p.setBrush(theme.color("accent"))
            hf = min(fi, c.length) / SR * self.pps
            hfo = min(fo, c.length) / SR * self.pps
            p.drawEllipse(QPointF(r.left() + hf, r.top() + 2), 4, 4)
            p.drawEllipse(QPointF(r.right() - hfo, r.top() + 2), 4, 4)
            f = QFont(self.font())
            f.setPixelSize(11)
            p.setFont(f)
            tag = c.name + (f"  {c.pitch:+.1f} st" if abs(c.pitch) > 0.01 else "") + ("  [LOCK]" if c.lock else "") + ("  [MUTE]" if c.mute else "") + ("  [LOOP]" if c.loop else "")
            p.setPen(theme.color("text"))
            p.drawText(QRectF(r.left() + 10, r.top() + 4, max(0, r.width() - 20), 14), Qt.AlignLeft | Qt.AlignVCenter, tag)
            p.setPen(QPen(theme.color("accent") if sel else theme.color("border"), 2 if sel else 1))
            p.setBrush(Qt.NoBrush)
            p.drawPath(clip_path)
            p.restore()

    def _auto_y(self, top, db):
        return top + 6 + (1 - (db - DB_MIN) / (DB_MAX - DB_MIN)) * (AUTO_H - 12)

    def _paint_auto(self, p, t, top):
        p.fillRect(QRectF(HEADER_W, top, self.width() - HEADER_W, AUTO_H), theme.color("card", 150))
        p.setPen(QPen(theme.color("grid"), 1))
        for db in (0, -20, -40):
            y = self._auto_y(top, db)
            p.drawLine(HEADER_W, int(y), self.width(), int(y))
        f = QFont(self.font())
        f.setPixelSize(10)
        p.setFont(f)
        p.setPen(theme.color("muted"))
        p.drawText(QPointF(HEADER_W + 6, top + 12), "Volume automation (double-click to add / remove points)")
        pts = sorted(t.auto_volume)
        if pts:
            path = QPainterPath()
            path.moveTo(HEADER_W, self._auto_y(top, pts[0][1]))
            for s, v in pts:
                path.lineTo(self.sx(s), self._auto_y(top, v))
            path.lineTo(self.width(), self._auto_y(top, pts[-1][1]))
            p.setPen(QPen(theme.color("accent"), 2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            p.setBrush(theme.color("accent"))
            for s, v in pts:
                p.drawEllipse(QPointF(self.sx(s), self._auto_y(top, v)), 5, 5)

    # ------------------------------------------------------------------ hit testing
    def _hit_clip(self, pos):
        i, ry = self.track_at(pos.y())
        if i < 0 or pos.x() < HEADER_W:
            return None
        t = self.p.tracks[i]
        top = self.tops()[0][i]
        if ry >= TRACK_H:
            return None
        for c in reversed(t.clips):
            r = self.clip_rect(t, c, top)
            if r.adjusted(-3, 0, 3, 0).contains(pos):
                zone = "body"
                eff_in = min(c.fade_in, c.length) / SR * self.pps
                eff_out = min(c.fade_out, c.length) / SR * self.pps
                if abs(pos.x() - (r.left() + eff_in)) < 8 and pos.y() < r.top() + 14:
                    zone = "fade_in"
                elif abs(pos.x() - (r.right() - eff_out)) < 8 and pos.y() < r.top() + 14:
                    zone = "fade_out"
                elif abs(pos.x() - r.left()) < 6:
                    zone = "trim_l"
                elif abs(pos.x() - r.right()) < 6:
                    zone = "trim_r"
                else:
                    gy = r.top() + (DB_GAIN_MAX - c.gain_db) / (DB_GAIN_MAX - DB_GAIN_MIN) * r.height()
                    if abs(pos.y() - gy) < 5:
                        zone = "gain"
                return i, t, c, r, zone
        return None

    def _hit_header(self, pos):
        if pos.x() >= HEADER_W or pos.y() < RULER_H:
            return None
        i, ry = self.track_at(pos.y())
        if i < 0:
            return None
        t = self.p.tracks[i]
        if 30 <= ry <= 50:
            for k, name in enumerate(("mute", "solo", "auto")):
                if 14 + k * 34 <= pos.x() <= 14 + k * 34 + 28:
                    return t, name
        if 52 <= ry <= 72:
            return t, "vol"
        if 70 <= ry <= 92:
            return t, "pan"
        return t, "name"

    def _bar_value(self, x):
        return min(1.0, max(0.0, (x - 42) / (HEADER_W - 42 - 74)))

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, e):
        pos = e.position()
        self.setFocus()
        if e.button() == Qt.RightButton:
            self._right_click(pos)
            return
        if e.button() != Qt.LeftButton:
            return
        if pos.y() < RULER_H and pos.x() >= HEADER_W:
            self.seekRequested.emit(max(0, self.tx(pos.x())))
            self.drag = {"kind": "seek"}
            return
        hh = self._hit_header(pos)
        if hh:
            self._header_press(hh, pos, e)
            return
        # automation lane
        i, ry = self.track_at(pos.y())
        if i >= 0 and ry >= TRACK_H and pos.x() >= HEADER_W:
            self._auto_press(self.p.tracks[i], i, pos)
            return
        hit = self._hit_clip(pos)
        if hit:
            i, t, c, r, zone = hit
            self.sel_track = t.id
            if e.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier):
                (self.sel_clips.discard if c.id in self.sel_clips else self.sel_clips.add)(c.id)
            elif c.id not in self.sel_clips:
                self.sel_clips = {c.id}
            if self.mode == "Spot" and zone == "body" and not c.lock:
                self.spotRequested.emit(c.id)
                self.selectionChanged.emit()
                self.update()
                return
            self.drag = {"kind": zone, "cid": c.id, "x": pos.x(), "y": pos.y(), "i": i, "moved": False,
                         "orig": {cid: self.p.find_clip(cid)[1].start for cid in self.sel_clips},
                         "orig_c": (c.start, c.offset, c.length, c.fade_in, c.fade_out, c.gain_db), "alt": bool(e.modifiers() & Qt.AltModifier)}
            self.selectionChanged.emit()
            self.update()
            return
        # empty space: rubber band / click sets the playhead
        self.sel_clips.clear()
        if i >= 0:
            self.sel_track = self.p.tracks[i].id
        self.range = None
        self.drag = {"kind": "band", "a": pos, "b": pos, "x": pos.x()}
        self.seekRequested.emit(max(0, self.tx(pos.x())))
        self.selectionChanged.emit()
        self.update()

    def _header_press(self, hh, pos, e):
        t, what = hh
        self.sel_track = t.id
        grp = [x for x in self.p.tracks if t.group and x.group == t.group] or [t]
        if what == "mute":
            new = not t.mute
            for x in grp:
                self.p.set_track(x.id, mute=new)
        elif what == "solo":
            new = not t.solo
            for x in grp:
                self.p.set_track(x.id, solo=new)
        elif what == "auto":
            t.show_auto = not t.show_auto
            self.p.changed()
        elif what in ("vol", "pan"):
            self.p.push_undo("hdr:" + what)
            self.drag = {"kind": what, "tid": t.id}
            self._header_drag(pos)
        self.selectionChanged.emit()
        self.update()

    def _was(self, t, attr):
        return not getattr(t, attr)

    def _header_drag(self, pos):
        d = self.drag
        t = self.p.track(d["tid"])
        v = self._bar_value(pos.x())
        grp = [x for x in self.p.tracks if t.group and x.group == t.group] or [t]
        for x in grp:
            if d["kind"] == "vol":
                x.gain_db = round(DB_MIN + v * (DB_MAX - DB_MIN), 1) if abs(v - (0 - DB_MIN) / (DB_MAX - DB_MIN)) > 0.015 else 0.0
            else:
                x.pan = round(v * 2 - 1, 2) if abs(v - 0.5) > 0.02 else 0.0
        self.update()

    def _auto_press(self, t, i, pos):
        top = self.tops()[0][i] + TRACK_H
        for k, (s, v) in enumerate(sorted(t.auto_volume)):
            if abs(self.sx(s) - pos.x()) < 8 and abs(self._auto_y(top, v) - pos.y()) < 8:
                self.drag = {"kind": "auto", "tid": t.id, "idx": k, "top": top}
                self.p.push_undo()
                return
        self.sel_track = t.id
        db = DB_MIN + (1 - (pos.y() - top - 6) / (AUTO_H - 12)) * (DB_MAX - DB_MIN)
        self.p.auto_add(t.id, "volume", self.snap(self.tx(pos.x())), round(min(DB_MAX, max(DB_MIN, db)), 1))
        k = sorted(t.auto_volume).index(next(q for q in t.auto_volume if abs(q[0] - self.snap(self.tx(pos.x()))) < 2))
        self.drag = {"kind": "auto", "tid": t.id, "idx": k, "top": top}
        self.selectionChanged.emit()

    def mouseDoubleClickEvent(self, e):
        pos = e.position()
        hh = self._hit_header(pos)
        if hh and hh[1] == "name":
            name, ok = QInputDialog.getText(self, "Track name", "Name:", text=hh[0].name)
            if ok and name.strip():
                self.p.set_track(hh[0].id, name=name.strip())
            return
        i, ry = self.track_at(pos.y())
        if i >= 0 and ry >= TRACK_H and pos.x() >= HEADER_W:
            t = self.p.tracks[i]
            top = self.tops()[0][i] + TRACK_H
            for k, (s, v) in enumerate(sorted(t.auto_volume)):
                if abs(self.sx(s) - pos.x()) < 8 and abs(self._auto_y(top, v) - pos.y()) < 8:
                    self.p.auto_remove(t.id, "volume", k)
                    return
        hit = self._hit_clip(pos)
        if hit:
            self.sel_clips = {hit[2].id}
            self.range = (hit[2].start, hit[2].end)
            self.selectionChanged.emit()
            self.update()

    def mouseMoveEvent(self, e):
        pos, d = e.position(), self.drag
        if d is None:
            hit = self._hit_clip(pos)
            if hit:
                z = hit[4]
                self.setCursor(Qt.SizeHorCursor if z in ("trim_l", "trim_r", "fade_in", "fade_out") else Qt.SizeVerCursor if z == "gain" else Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return
        k = d["kind"]
        if k == "seek":
            self.seekRequested.emit(max(0, self.tx(pos.x())))
        elif k in ("vol", "pan"):
            self._header_drag(pos)
        elif k == "auto":
            t = self.p.track(d["tid"])
            db = DB_MIN + (1 - (pos.y() - d["top"] - 6) / (AUTO_H - 12)) * (DB_MAX - DB_MIN)
            self.p.auto_move(t.id, "volume", d["idx"], self.snap(self.tx(pos.x())), round(min(DB_MAX, max(DB_MIN, db)), 1))
            d["idx"] = min(d["idx"], len(t.auto_volume) - 1)
            self.update()
        elif k == "band":
            d["b"] = pos
            a = self.tx(min(d["a"].x(), pos.x()))
            b = self.tx(max(d["a"].x(), pos.x()))
            self.range = (max(0, a), max(0, b))
            r = QRectF(d["a"], pos).normalized()
            tops, _ = self.tops()
            self.sel_clips = {c.id for i, t in enumerate(self.p.tracks) for c in t.clips if self.clip_rect(t, c, tops[i]).intersects(r)}
            self.selectionChanged.emit()
            self.update()
        else:
            self._clip_drag(pos, d, e)

    def _clip_drag(self, pos, d, e):
        dx = pos.x() - d["x"]
        if not d["moved"] and abs(dx) < 3 and abs(pos.y() - d["y"]) < 3:
            return
        if not d["moved"]:
            self.p.push_undo()
            d["moved"] = True
        t, c = self.p.find_clip(d["cid"])
        if c is None:
            return
        k = d["kind"]
        dsamp = int(dx / self.pps * SR)
        if k == "body":
            base = d["orig"][c.id] + dsamp
            delta = (self.snap(base) if (self.mode == "Grid" or self.grid > 0) else base) - d["orig"][c.id]
            ti = self.track_at(pos.y())[0]
            dt = (ti - d["i"]) if ti >= 0 else 0
            for cid, st in d["orig"].items():
                tt, cc = self.p.find_clip(cid)
                if cc is None or cc.lock:
                    continue
                cc.start = max(0, st + delta)
                if dt and len(d["orig"]) == 1 and 0 <= d["i"] + dt < len(self.p.tracks):
                    new_t = self.p.tracks[d["i"] + dt]
                    if new_t.id != tt.id:
                        tt.clips.remove(cc)
                        new_t.clips.append(cc)
                        d["i"] = d["i"] + dt
                        d["y"] = pos.y()
        elif k == "trim_l":
            self.p.trim_start(c.id, self.snap(d["orig_c"][0] + dsamp))
        elif k == "trim_r":
            self.p.trim_end(c.id, self.snap(d["orig_c"][0] + d["orig_c"][2] + dsamp), allow_loop=d["alt"])
        elif k == "fade_in":
            c.fade_in = int(min(max(0, d["orig_c"][3] + dsamp), c.length))
        elif k == "fade_out":
            c.fade_out = int(min(max(0, d["orig_c"][4] - dsamp), c.length))
        elif k == "gain":
            r = self.clip_rect(t, c, self.tops()[0][d["i"]])
            g = DB_GAIN_MAX - (pos.y() - r.top()) / r.height() * (DB_GAIN_MAX - DB_GAIN_MIN)
            c.gain_db = round(min(DB_GAIN_MAX, max(DB_GAIN_MIN, g)) * 2) / 2
        self.update()

    def mouseReleaseEvent(self, e):
        d, self.drag = self.drag, None
        if not d:
            return
        k = d["kind"]
        if k in ("band", "seek"):
            self.update()
            return
        if k == "body" and d.get("moved") and self.mode == "Shuffle":
            for tr in self.p.tracks:
                self.p._repack(tr)
        if k in ("auto", "vol", "pan") or d.get("moved"):
            self.p.changed()
        self.selectionChanged.emit()
        self.update()

    def _right_click(self, pos):
        hit = self._hit_clip(pos)
        if hit and hit[2].id not in self.sel_clips:
            self.sel_clips = {hit[2].id}
            self.sel_track = hit[1].id
            self.selectionChanged.emit()
        elif not hit:
            i = self.track_at(pos.y())[0]
            if i >= 0:
                self.sel_track = self.p.tracks[i].id
        self.update()
        self.contextRequested.emit(self.mapToGlobal(pos.toPoint()))

    def wheelEvent(self, e):
        dy = e.angleDelta().y()
        if e.modifiers() & Qt.ControlModifier:
            self.zoom(1.25 if dy > 0 else 0.8, e.position().x())
        elif e.modifiers() & Qt.ShiftModifier:
            self.set_t0(self.t0 - dy / 120 * 80 / self.pps)
        else:
            _, total = self.tops()
            self.y0 = int(min(max(0, total + self.y0 - self.height() + 20), max(0, self.y0 - dy / 120 * 40)))
            self.update()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        pos = e.position()
        self.filesDropped.emit(paths, self.track_at(pos.y())[0], max(0, self.snap(self.tx(max(pos.x(), HEADER_W)))))


DB_GAIN_MIN, DB_GAIN_MAX = -24.0, 12.0


# ======================================================================== editor container
class TimelineEditor(QWidget):
    def __init__(self, project, session, store, main):
        super().__init__()
        self.p, self.s, self.store, self.main = project, session, store, main
        self.view = TimelineView(project, session)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addLayout(self._toolbar())
        v.addWidget(self.view, 1)
        self.bar = QScrollBar(Qt.Horizontal)
        self.bar.valueChanged.connect(lambda val: self.view.set_t0(val / 100.0) if not self._busy else None)
        v.addWidget(self.bar)
        self._busy = False
        self.view.selectionChanged.connect(self._sync_bar)
        project.add_listener(self._sync_bar)
        self.view.seekRequested.connect(lambda g: (self._ensure_attached(), session.seek_global(g)))
        self.view.filesDropped.connect(self.import_files)
        self.view.spotRequested.connect(self.spot)
        self.view.contextRequested.connect(self.context_menu)
        project.add_listener(self._changed)

    # ------------------------------------------------------------------ toolbar
    def _combo(self, items, tip_text, width=86):
        c = QComboBox()
        c.addItems(items)
        c.setMinimumWidth(width)
        tip(c, tip_text)
        return c

    def _btn(self, text, fn, tip_text):
        b = QPushButton(text)
        b.clicked.connect(lambda _=False: fn())
        tip(b, tip_text)
        return b

    def _menu_btn(self, text, tip_text, entries):
        b = QToolButton()
        b.setText(text + "  v")
        b.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(b)
        for ent in entries:
            if ent is None:
                m.addSeparator()
            else:
                a = m.addAction(ent[0])
                a.triggered.connect(lambda _=False, fn=ent[1]: fn())
                if len(ent) > 2:
                    a.setToolTip(ent[2])
        b.setMenu(m)
        tip(b, tip_text)
        return b

    def _toolbar(self):
        col = QVBoxLayout()
        r1, r2 = QHBoxLayout(), QHBoxLayout()
        col.addLayout(r1)
        col.addLayout(r2)
        self.mode = self._combo(EDIT_MODES, "Edit mode: Slip = free movement. Shuffle = clips close up gaps. Grid = clips snap to the grid. Spot = click a clip to type its exact start time.")
        self.mode.currentTextChanged.connect(lambda m: (setattr(self.view, "mode", m), self.view.update()))
        self.grid = self._combo(list(GRID_CHOICES) + ["1/4 note", "1/8 note", "1/16 note", "1 bar"], "Grid value: the spacing clips snap to in Grid mode and for 'Separate at grid'. Note values follow the session tempo.", 90)
        self.grid.currentTextChanged.connect(lambda g: self.apply_grid())
        self.nudge = self._combo(list(NUDGE_CHOICES), "Nudge value: how far the nudge buttons move the selected clips.", 80)
        self.nudge.setCurrentText("10 ms")
        for w, lab in ((self.mode, "Mode"), (self.grid, "Grid"), (self.nudge, "Nudge")):
            r1.addWidget(QLabel(lab))
            r1.addWidget(w)
        r1.addWidget(self._btn("<", lambda: self.do_nudge(-1), "Nudge selected clips earlier by the nudge value."))
        r1.addWidget(self._btn(">", lambda: self.do_nudge(1), "Nudge selected clips later by the nudge value."))
        r1.addSpacing(10)
        r1.addWidget(self._btn("Add Track", self.add_track, "Add an empty track."))
        r1.addWidget(self._btn("Add Bus", self.add_bus, "Add a submix bus: route tracks to it, or send to it for shared reverb and delay."))
        r1.addWidget(self._btn("Import...", self.import_dialog, "Import audio or video files onto new tracks."))
        r1.addWidget(self._btn("Stems...", self.render_stems, "Render every track to its own WAV file (all the same length so they line up)."))
        r1.addStretch(1)
        self.auto_prev = QCheckBox("Auto-update cleaned preview (B)")
        tip(self.auto_prev, "When on, the master cleanup preview re-renders after every edit. Leave off for fast editing and click Render cleaned preview when ready.")
        self.auto_prev.toggled.connect(self.s.set_auto_preview)
        r1.addWidget(self.auto_prev)
        r1.addWidget(self._btn("Render cleaned preview", lambda: self.s.request_preview(), "Run the master cleanup (your FastTrack/Precision settings) on the whole mix so you can compare A and B."))
        for txt, fn, tt in (("Cut", self.cut, "Cut: copy the selected clips and remove them."), ("Copy", self.copy, "Copy the selected clips."),
                            ("Paste", self.paste, "Paste at the playhead on the selected track."), ("Clear", self.clear, "Clear: delete the selected clips (Delete key)."),
                            ("Duplicate", self.duplicate, "Duplicate: place a copy right after the selected clips."), ("Repeat...", self.repeat, "Repeat: place several copies back to back.")):
            r2.addWidget(self._btn(txt, fn, tt))
        r2.addWidget(self._menu_btn("Separate", "Split clips into pieces.", [
            ("At playhead", self.sep_playhead, "Separate selected clips (or all clips) where the playhead is."),
            ("At selection range", self.sep_selection, "Separate at the start and end of the selected time range."),
            ("At grid", self.sep_grid, "Separate at every grid line."), ("At transients", self.sep_transients, "Separate where sounds suddenly start (words, hits).")]))
        r2.addWidget(self._menu_btn("Trim", "Shorten or lengthen clips.", [
            ("Start to playhead", lambda: self.trim_playhead("start"), "Standard trim: start the clip at the playhead."),
            ("End to playhead", lambda: self.trim_playhead("end"), "Standard trim: end the clip at the playhead."),
            ("To selection range", self.trim_range, "Keep only the selected time range."),
            ("Loop-trim end to playhead", lambda: self.trim_playhead("loop"), "Loop trim: extend the clip to the playhead by repeating it.")]))
        r2.addWidget(self._menu_btn("Fades", "Fade clips in and out.", [
            ("Fade in to playhead", lambda: self.fade_playhead(True), "Fade in from the clip start to the playhead."),
            ("Fade out from playhead", lambda: self.fade_playhead(False), "Fade out from the playhead to the clip end."), None,
            ("Batch fades...", self.batch_fades, "Apply the same fade in/out length and curve to all selected clips."),
            ("Curve: Equal power", lambda: self.set_curve("equal_power"), "Constant-power curve: best for crossfading different sounds."),
            ("Curve: Linear", lambda: self.set_curve("linear"), "Straight-line fade."),
            ("Curve: Equal gain", lambda: self.set_curve("equal_gain"), "Constant-gain curve: best for crossfading identical sounds.")]))
        r2.addWidget(self._menu_btn("Clip", "Clip properties.", [
            ("Clip gain...", self.clip_gain, "Raise or lower the volume of the selected clips (also drag the dashed line)."),
            ("Normalize clip", self.clip_normalize, "Raise each selected clip so its peak reaches -1 dB."),
            ("Mute / unmute", self.clip_mute, "Silence the selected clips without deleting them."),
            ("Lock / unlock", self.clip_lock, "Lock clips so they cannot be moved, trimmed or deleted."),
            ("Pitch shift...", self.clip_pitch, "Change the pitch of the selected clips in semitones, keeping their length."),
            ("Spot to time...", self.spot_selected, "Type the exact start time of the selected clip.")]))
        r2.addStretch(1)
        for txt, fn, tt in (("Fit", self.view.zoom_fit, "Show the whole project."), ("+", lambda: self.view.zoom(1.4), "Zoom in (Ctrl + wheel)."),
                            ("-", lambda: self.view.zoom(0.7), "Zoom out (Ctrl + wheel).")):
            r2.addWidget(self._btn(txt, fn, tt))
        return col

    def _grid_seconds(self, label):
        if label in GRID_CHOICES:
            return GRID_CHOICES[label]
        beat = 60.0 / max(self.p.settings.get("tempo", 120.0), 1)
        return {"1/4 note": beat, "1/8 note": beat / 2, "1/16 note": beat / 4, "1 bar": beat * int(self.p.settings.get("sig_num", 4))}[label]

    def apply_grid(self):
        self.view.grid = self._grid_seconds(self.grid.currentText())
        self.view.update()

    def select_all(self):
        self.view.sel_clips = set(self._all_ids())
        self.view.selectionChanged.emit()
        self.view.update()

    def fade_all(self, ms):
        ids = self.ids() or self._all_ids()
        n = int(ms / 1000 * SR)
        self.p.set_fades(ids, n, n)

    def zoom_fit_view(self):
        self.view.zoom_fit()

    def _sync_bar(self):
        self._busy = True
        self.bar.setRange(0, int(self.view.total_seconds() * 100))
        self.bar.setPageStep(int((self.view.width() - HEADER_W) / self.view.pps * 100))
        self.bar.setValue(int(self.view.t0 * 100))
        self._busy = False

    # ------------------------------------------------------------------ project <-> session
    def _changed(self):
        self.view.update()

    def _ensure_attached(self):
        self.main.attach_project()

    def ensure_loaded(self):
        """Opening a file then switching to the timeline shows it as Track 1 (nothing is lost)."""
        s = self.s
        if not self.p.tracks and s.audio is not None and not s.project_mode:
            name = Path(s.src_path).stem if s.src_path else "Audio"
            self.p.import_array(name, s.audio.copy(), s.src_path or "")
            self.p.undo_stack.clear()
            self.main.attach_project()
            self.view.zoom_fit()
            self.view.sel_track = self.p.tracks[0].id

    # ------------------------------------------------------------------ helpers
    def ids(self, fallback_all=False):
        ids = list(self.view.sel_clips)
        if not ids and self.view.sel_track:
            t = self.p.track(self.view.sel_track)
            if t and fallback_all:
                ids = [c.id for c in t.clips]
        return ids

    def need(self, ids):
        if not ids:
            self.main.statusBar().showMessage("Select one or more clips first (click a clip, or drag a box around several).", 4000)
        return bool(ids)

    def ph(self):
        return int(self.s.global_pos())

    def _sel_track_index(self):
        t = self.p.track(self.view.sel_track) if self.view.sel_track else None
        return self.p.tracks.index(t) if t in self.p.tracks else 0

    # ------------------------------------------------------------------ track / import
    def add_track(self):
        self._ensure_attached()
        t = self.p.add_track()
        self.view.sel_track = t.id

    def add_bus(self):
        self._ensure_attached()
        t = self.p.add_bus()
        self.view.sel_track = t.id

    def import_dialog(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Import audio or video", "", "Audio / video (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.aif *.aiff *.mp4 *.mov *.mkv *.avi *.mts *.mxf *.webm);;All files (*.*)")
        if fs:
            self.import_files(fs, -1, 0)

    def import_files(self, paths, track_index=-1, start=0):
        self._ensure_attached()
        for path in paths:
            if Path(path).is_dir():
                continue
            self.main.statusBar().showMessage(f"Importing {Path(path).name}...")
            t = Task(lambda prog, cancel, path=path: engine.decode(path, cancel))
            self.s._hold(t)
            tid = self.p.tracks[track_index].id if 0 <= track_index < len(self.p.tracks) else None
            t.sig.done.connect(lambda data, path=path, tid=tid, start=start: self._imported(path, data, tid, start))
            t.sig.failed.connect(lambda m, path=path: QMessageBox.warning(self, "Import", f"Could not import {Path(path).name}:\n{m}"))
            self.s.pool.start(t)

    def _imported(self, path, data, tid, start):
        self.p.import_array(Path(path).name, data, path, track_id=tid, start=start)
        self.main.statusBar().showMessage("Imported " + Path(path).name, 3000)
        self.view.zoom_fit() if len(self.p.tracks) <= 1 else None

    def render_stems(self):
        if not self.p.tracks:
            return
        d = QFileDialog.getExistingDirectory(self, "Choose a folder for the stems")
        if not d:
            return
        out = Path(d) / f"{self.p.name} stems"
        self.main.statusBar().showMessage("Rendering stems...")
        t = Task(lambda prog, cancel: self.p.render_stems(out))
        self.s._hold(t)
        t.sig.done.connect(lambda paths: QMessageBox.information(self, "Stems", f"Saved {len(paths)} stems to:\n{out}"))
        t.sig.failed.connect(lambda m: QMessageBox.warning(self, "Stems", f"Stem export failed:\n{m}"))
        self.s.pool.start(t)

    # ------------------------------------------------------------------ edits
    def cut(self):
        ids = self.ids()
        if self.need(ids):
            self.p.copy_clips(ids)
            self.p.delete(ids, self.view.mode == "Shuffle")
            self.view.sel_clips.clear()
            self.view.selectionChanged.emit()

    def copy(self):
        ids = self.ids()
        if self.need(ids):
            self.p.copy_clips(ids)
            self.main.statusBar().showMessage(f"Copied {len(ids)} clip(s)", 2500)

    def paste(self):
        if not self.p.clipboard:
            self.main.statusBar().showMessage("Nothing to paste: copy some clips first.", 3000)
            return
        new = self.p.paste(self._sel_track_index(), self.snap_ph())
        self.view.sel_clips = set(new)
        self.view.selectionChanged.emit()

    def snap_ph(self):
        return self.view.snap(self.ph())

    def clear(self):
        ids = self.ids()
        if self.need(ids):
            self.p.delete(ids, self.view.mode == "Shuffle")
            self.view.sel_clips.clear()
            self.view.selectionChanged.emit()

    def duplicate(self):
        ids = self.ids()
        if self.need(ids):
            self.view.sel_clips = set(self.p.duplicate(ids))
            self.view.selectionChanged.emit()

    def repeat(self):
        ids = self.ids()
        if not self.need(ids):
            return
        n, ok = QInputDialog.getInt(self, "Repeat", "Number of repeats:", 3, 1, 200)
        if ok:
            self.view.sel_clips = set(self.p.repeat(ids, n))
            self.view.selectionChanged.emit()

    def do_nudge(self, direction):
        ids = self.ids()
        if self.need(ids):
            self.p.nudge(ids, direction * int(NUDGE_CHOICES[self.nudge.currentText()] * SR))

    def _all_ids(self):
        return [c.id for t in self.p.tracks for c in t.clips]

    def sep_playhead(self):
        ids = self.ids() or self._all_ids()
        self.view.sel_clips = set(self.p.split(ids, self.snap_ph()) + list(self.view.sel_clips))
        self.view.selectionChanged.emit()

    def sep_selection(self):
        r = self.view.range
        if not r:
            self.main.statusBar().showMessage("Drag across the timeline to choose a time range first.", 4000)
            return
        ids = self.ids() or self._all_ids()
        self.p.split(ids, r[0])
        self.p.split(self.ids() or self._all_ids(), r[1])

    def sep_grid(self):
        if self.view.grid <= 0:
            self.main.statusBar().showMessage("Choose a Grid value first (not Off).", 4000)
            return
        ids = self.ids() or self._all_ids()
        self.p.separate_at_grid(ids, self.view.grid)

    def sep_transients(self):
        ids = self.ids() or self._all_ids()
        n = self.p.separate_at_transients(ids)
        self.main.statusBar().showMessage(f"Separated at {n} transient(s).", 4000)

    def trim_playhead(self, which):
        ids = self.ids()
        if not self.need(ids):
            return
        ph = self.snap_ph()
        self.p.push_undo()
        for cid in ids:
            if which == "start":
                self.p.trim_start(cid, ph)
            else:
                self.p.trim_end(cid, ph, allow_loop=(which == "loop"))
        self.p.changed()

    def trim_range(self):
        r = self.view.range
        ids = self.ids() or self._all_ids()
        if not r:
            self.main.statusBar().showMessage("Drag across the timeline to choose a time range first.", 4000)
            return
        self.p.trim_to(ids, r[0], r[1])

    def fade_playhead(self, fade_in):
        ids = self.ids()
        if not self.need(ids):
            return
        ph = self.snap_ph()
        for cid in ids:
            t, c = self.p.find_clip(cid)
            if c:
                self.p.set_fades([cid], fade_in=ph - c.start if fade_in else None, fade_out=c.end - ph if not fade_in else None)

    def batch_fades(self):
        ids = self.ids()
        if not self.need(ids):
            return
        ms, ok = QInputDialog.getInt(self, "Batch fades", "Fade length (ms) applied to the start and end of every selected clip:", 20, 0, 10000)
        if not ok:
            return
        curve, ok = QInputDialog.getItem(self, "Batch fades", "Curve:", ["equal_power", "linear", "equal_gain"], 0, False)
        if ok:
            n = int(ms / 1000 * SR)
            self.p.set_fades(ids, n, n, curve, curve)

    def set_curve(self, curve):
        ids = self.ids()
        if self.need(ids):
            self.p.set_fades(ids, in_curve=curve, out_curve=curve)

    def clip_gain(self):
        ids = self.ids()
        if not self.need(ids):
            return
        t, c = self.p.find_clip(ids[0])
        db, ok = QInputDialog.getDouble(self, "Clip gain", "Gain (dB):", c.gain_db, -24, 12, 1)
        if ok:
            self.p.set_clip(ids, gain_db=db)

    def clip_normalize(self):
        ids = self.ids()
        if not self.need(ids):
            return
        self.p.push_undo()
        for cid in ids:
            t, c = self.p.find_clip(cid)
            seg = self.p._segment(c)
            pk = float(np.abs(seg).max())
            if pk > 1e-6:
                c.gain_db = round(float(np.clip(-1.0 - 20 * math.log10(pk), -24, 12)), 1)
        self.p.changed()

    def clip_mute(self):
        ids = self.ids()
        if self.need(ids):
            t, c = self.p.find_clip(ids[0])
            self.p.set_clip(ids, mute=not c.mute)

    def clip_lock(self):
        ids = self.ids()
        if self.need(ids):
            t, c = self.p.find_clip(ids[0])
            self.p.set_clip(ids, lock=not c.lock)

    def clip_pitch(self):
        ids = self.ids()
        if not self.need(ids):
            return
        t, c = self.p.find_clip(ids[0])
        st, ok = QInputDialog.getDouble(self, "Pitch shift", "Semitones (-12 to +12):", c.pitch, -12, 12, 1)
        if ok:
            self.main.statusBar().showMessage("Shifting pitch...", 3000)
            self.p.set_clip(ids, pitch=st)

    def spot_selected(self):
        ids = self.ids()
        if self.need(ids):
            self.spot(ids[0])

    def spot(self, cid):
        t, c = self.p.find_clip(cid)
        if not c:
            return
        txt, ok = QInputDialog.getText(self, "Spot", "Start time (m:ss.ms or seconds):", text=f"{c.start / SR:.3f}")
        if not ok:
            return
        try:
            if ":" in txt:
                m, s = txt.split(":")
                sec = int(m) * 60 + float(s)
            else:
                sec = float(txt)
        except ValueError:
            return
        self.p.push_undo()
        c.start = max(0, int(sec * SR))
        self.p.changed()

    def context_menu(self, gpos):
        m = QMenu(self)
        for txt, fn in (("Cut", self.cut), ("Copy", self.copy), ("Paste", self.paste), ("Clear", self.clear), None,
                        ("Duplicate", self.duplicate), ("Repeat...", self.repeat), ("Separate at playhead", self.sep_playhead), None,
                        ("Clip gain...", self.clip_gain), ("Mute / unmute", self.clip_mute), ("Lock / unlock", self.clip_lock), ("Pitch shift...", self.clip_pitch)):
            if txt is None:
                m.addSeparator()
            else:
                m.addAction(txt, fn)
        m.exec(gpos)


# ======================================================================== inspector
class _Param(QWidget):
    def __init__(self, lo, hi, step, fmt, on_change):
        super().__init__()
        l = QHBoxLayout(self)
        l.setContentsMargins(0, 0, 0, 0)
        self.sl = QSlider(Qt.Horizontal)
        self.sl.setRange(int(lo / step), int(hi / step))
        self.lab = QLabel()
        self.lab.setMinimumWidth(66)
        self.lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        l.addWidget(self.sl, 1)
        l.addWidget(self.lab)
        self.step, self.fmt, self.cb = step, fmt, on_change
        self.sl.valueChanged.connect(self._moved)

    def _moved(self, v):
        self.lab.setText(self.fmt.format(v * self.step))
        self.cb(v * self.step)

    def set(self, val):
        self.sl.blockSignals(True)
        self.sl.setValue(int(round(val / self.step)))
        self.sl.blockSignals(False)
        self.lab.setText(self.fmt.format(self.sl.value() * self.step))


class TrackInspector(QWidget):
    """Track and clip properties: fader, pan, low cut, 3-band EQ, gate and compression per track; clip gain, fades, pitch."""

    def __init__(self, project, editor):
        super().__init__()
        self.p, self.ed = project, editor
        v = QVBoxLayout(self)
        self.tbox = QGroupBox("Selected track")
        f = QFormLayout(self.tbox)
        self.name = QLabel("-")
        f.addRow("Name", self.name)
        T = lambda key: (lambda val, key=key: self._set_track(key, val))
        self.t_gain = _Param(-60, 12, 0.5, "{:+.1f} dB", T("gain_db"))
        self.t_pan = _Param(-1, 1, 0.01, "{:+.2f}", T("pan"))
        self.t_low = _Param(0, 400, 5, "{:.0f} Hz", T("lowcut"))
        self.t_eql = _Param(-12, 12, 0.5, "{:+.1f} dB", T("eq_low"))
        self.t_eqm = _Param(-12, 12, 0.5, "{:+.1f} dB", T("eq_mid"))
        self.t_eqh = _Param(-12, 12, 0.5, "{:+.1f} dB", T("eq_high"))
        self.t_gate = _Param(-80, -20, 1, "{:.0f} dB", T("gate_thr"))
        self.t_comp = _Param(0, 1, 0.05, "{:.0%}", T("comp"))
        self.group = QComboBox()
        self.group.addItems(["None"] + [f"Group {i}" for i in range(1, 9)])
        tip(self.group, "Tracks in the same group share mute, solo and fader moves.")
        self.group.currentIndexChanged.connect(lambda i: self._set_track("group", i))
        for lab, w, tt in (("Volume", self.t_gain, "Track fader."), ("Pan", self.t_pan, "Left/right balance."), ("Low cut", self.t_low, "Remove rumble below this frequency."),
                           ("EQ low", self.t_eql, "Boost or cut bass (below 150 Hz)."), ("EQ mid", self.t_eqm, "Boost or cut the mids (around 1.5 kHz)."),
                           ("EQ high", self.t_eqh, "Boost or cut treble (above 6 kHz)."), ("Gate", self.t_gate, "Lower quiet sound below this level. -80 = off."),
                           ("Compress", self.t_comp, "Even out loud and quiet sound.")):
            tip(w, tt)
            f.addRow(lab, w)
        f.addRow("Group", self.group)
        v.addWidget(self.tbox)
        self.cbox = QGroupBox("Selected clip")
        g = QFormLayout(self.cbox)
        self.cname = QLabel("-")
        g.addRow("Name", self.cname)
        C = lambda key: (lambda val, key=key: self._set_clip(key, val))
        self.c_gain = _Param(-24, 12, 0.5, "{:+.1f} dB", C("gain_db"))
        self.c_fi = _Param(0, 3000, 5, "{:.0f} ms", lambda val: self._set_fade("in", val))
        self.c_fo = _Param(0, 3000, 5, "{:.0f} ms", lambda val: self._set_fade("out", val))
        self.c_pitch = _Param(-12, 12, 0.5, "{:+.1f} st", C("pitch"))
        for lab, w, tt in (("Gain", self.c_gain, "Clip gain."), ("Fade in", self.c_fi, "Fade-in length."), ("Fade out", self.c_fo, "Fade-out length."),
                           ("Pitch", self.c_pitch, "Pitch shift in semitones, keeping length.")):
            tip(w, tt)
            g.addRow(lab, w)
        self.curve_in = QComboBox()
        self.curve_in.addItems(FADE_CURVES)
        self.curve_out = QComboBox()
        self.curve_out.addItems(FADE_CURVES)
        self.curve_in.currentTextChanged.connect(lambda c: self._clips(lambda ids: self.p.set_fades(ids, in_curve=c)))
        self.curve_out.currentTextChanged.connect(lambda c: self._clips(lambda ids: self.p.set_fades(ids, out_curve=c)))
        g.addRow("In curve", self.curve_in)
        g.addRow("Out curve", self.curve_out)
        v.addWidget(self.cbox)
        v.addStretch(1)
        editor.view.selectionChanged.connect(self.refresh)
        project.add_listener(self.refresh)
        self.refresh()

    def _clips(self, fn):
        ids = list(self.ed.view.sel_clips)
        if ids:
            fn(ids)

    def _set_track(self, key, val):
        t = self.p.track(self.ed.view.sel_track)
        if t:
            val = int(val) if key == "group" else float(val)
            self.p.set_track(t.id, **{key: val})

    def _set_clip(self, key, val):
        self._clips(lambda ids: self.p.set_clip(ids, **{key: float(val)}))

    def _set_fade(self, which, ms):
        n = int(ms / 1000 * SR)
        self._clips(lambda ids: self.p.set_fades(ids, **({"fade_in": n} if which == "in" else {"fade_out": n})))

    def refresh(self):
        t = self.p.track(self.ed.view.sel_track) if self.ed.view.sel_track else None
        self.tbox.setEnabled(t is not None)
        if t:
            self.name.setText(t.name)
            for w, val in ((self.t_gain, t.gain_db), (self.t_pan, t.pan), (self.t_low, t.lowcut), (self.t_eql, t.eq_low), (self.t_eqm, t.eq_mid),
                           (self.t_eqh, t.eq_high), (self.t_gate, t.gate_thr), (self.t_comp, t.comp)):
                w.set(val)
            self.group.blockSignals(True)
            self.group.setCurrentIndex(t.group)
            self.group.blockSignals(False)
        ids = list(self.ed.view.sel_clips)
        c = self.p.find_clip(ids[0])[1] if ids else None
        self.cbox.setEnabled(c is not None)
        if c:
            self.cname.setText(c.name + (f"  (+{len(ids) - 1} more)" if len(ids) > 1 else ""))
            self.c_gain.set(c.gain_db)
            self.c_fi.set(c.fade_in / SR * 1000)
            self.c_fo.set(c.fade_out / SR * 1000)
            self.c_pitch.set(c.pitch)
            for cb, val in ((self.curve_in, c.fade_in_curve), (self.curve_out, c.fade_out_curve)):
                cb.blockSignals(True)
                cb.setCurrentText(val)
                cb.blockSignals(False)
