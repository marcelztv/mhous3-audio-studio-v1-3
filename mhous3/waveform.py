"""Waveform / spectrogram view.  Original = grey, cleaned = amber overlay."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QScrollBar, QVBoxLayout, QWidget

from . import theme

SR = 48000
_BLOCK = 256


def _mip(mono):
    n = len(mono) // _BLOCK * _BLOCK
    if n == 0:
        return None
    b = mono[:n].reshape(-1, _BLOCK)
    return b.min(axis=1), b.max(axis=1)


def _make_lut(stops):
    xs = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(xs, [t[0] for t in stops], [t[1][c] for t in stops])
    return lut


_LUTS = {
    "dark": _make_lut([(0.0, (12, 14, 20)), (0.25, (22, 50, 96)), (0.5, (40, 120, 200)), (0.75, (110, 200, 255)), (1.0, (255, 255, 255))]),
    "light": _make_lut([(0.0, (250, 252, 255)), (0.25, (200, 225, 250)), (0.5, (110, 170, 235)), (0.75, (35, 110, 200)), (1.0, (10, 40, 110))]),
}


def spec_lut():
    return _LUTS[theme.current_name()]


class WaveformView(QWidget):
    seekRequested = Signal(int)
    selectionChanged = Signal()
    viewChanged = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.orig = self.clean = None
        self._mo = self._mc = None
        self.clean_off = 0
        self.vs, self.vl = 0, SR * 10
        self.sel = None
        self.playhead = 0
        self.mode = "wave"
        self.autoscale = True
        self._drag = None
        self._spec_cache = (None, None)

    # ----- data
    @property
    def n(self):
        return 0 if self.orig is None else len(self.orig)

    def set_orig(self, stereo, keep_view=False):
        if stereo is None:
            self.orig = self._mo = self.clean = None
        else:
            self.orig = stereo.mean(axis=1).astype(np.float32)
            self._mo = _mip(self.orig)
        self.clean = self._mc = None
        self.sel = None
        if not keep_view:
            self.vs, self.vl = 0, max(self.n, 1)
        else:
            self._clamp()
        self._spec_cache = (None, None)
        self.viewChanged.emit()
        self.update()

    def set_clean(self, stereo, offset=0):
        if stereo is None:
            self.clean = self._mc = None
        else:
            self.clean = stereo.mean(axis=1).astype(np.float32)
            self._mc = _mip(self.clean)
            self.clean_off = int(offset)
        self.update()

    def set_selection(self, sel):
        if sel != self.sel:
            self.sel = sel
            self.update()

    def set_playhead(self, g):
        if g != self.playhead:
            self.playhead = g
            self.update()

    def set_mode(self, mode):
        self.mode = mode
        self.update()

    # ----- view control
    def _clamp(self):
        self.vl = int(max(64, min(self.vl, max(self.n, 64))))
        self.vs = int(max(0, min(self.vs, max(self.n - self.vl, 0))))

    def set_view(self, start, length=None):
        self.vs = start
        if length is not None:
            self.vl = length
        self._clamp()
        self.viewChanged.emit()
        self.update()

    def zoom_fit(self):
        self.set_view(0, self.n)

    def zoom(self, factor, center=None):
        c = self.vs + self.vl / 2 if center is None else center
        newl = max(64, min(self.vl * factor, max(self.n, 64)))
        self.set_view(int(c - newl * (c - self.vs) / self.vl), int(newl))

    def zoom_to_selection(self):
        if self.sel and self.sel[1] - self.sel[0] > 64:
            self.set_view(self.sel[0], self.sel[1] - self.sel[0])

    def ensure_visible(self, pos):
        if pos < self.vs or pos > self.vs + self.vl:
            self.set_view(int(pos - self.vl * 0.1))

    def x_to_sample(self, x):
        return int(self.vs + x / max(self.width(), 1) * self.vl)

    def sample_to_x(self, s):
        return (s - self.vs) / self.vl * self.width()

    # ----- interaction
    def mousePressEvent(self, e):
        if self.orig is None or e.button() != Qt.LeftButton:
            return
        self._drag = (e.position().x(), self.x_to_sample(e.position().x()), False)

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        x0, s0, moved = self._drag
        if abs(e.position().x() - x0) > 3 or moved:
            s1 = max(0, min(self.n, self.x_to_sample(e.position().x())))
            self.sel = (min(s0, s1), max(s0, s1))
            self._drag = (x0, s0, True)
            self.selectionChanged.emit()
            self.update()

    def mouseReleaseEvent(self, e):
        if self._drag is None:
            return
        x0, s0, moved = self._drag
        self._drag = None
        if not moved:
            self.sel = None
            self.selectionChanged.emit()
            self.seekRequested.emit(max(0, min(self.n, s0)))
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.sel = (0, self.n) if self.orig is not None else None
        self.selectionChanged.emit()
        self.update()

    def wheelEvent(self, e):
        if self.orig is None:
            return
        dy = e.angleDelta().y()
        if e.modifiers() & Qt.ShiftModifier:
            self.set_view(int(self.vs - dy / 120 * self.vl * 0.1))
        else:
            self.zoom(0.8 if dy > 0 else 1.25, self.x_to_sample(e.position().x()))

    # ----- painting
    def _cols(self, mn, mx, scale, off, w):
        edges = np.linspace(self.vs, self.vs + self.vl, w + 1) / scale - off / scale
        lo = np.full(w, np.nan, np.float32)
        hi = np.full(w, np.nan, np.float32)
        valid = (edges[1:] > 0) & (edges[:-1] < len(mn))
        if not valid.any():
            return lo, hi
        e = np.clip(edges, 0, len(mn)).astype(np.int64)
        starts = e[:-1][valid]
        seg0, seg1 = int(starts.min()), int(max(e[1:][valid].max(), starts.min() + 1))
        idx = np.minimum(starts - seg0, seg1 - seg0 - 1)
        lo[valid] = np.minimum.reduceat(mn[seg0:seg1], idx)
        hi[valid] = np.maximum.reduceat(mx[seg0:seg1], idx)
        return lo, hi

    def _wave(self, arr, mip, off, w):
        spp = self.vl / max(w, 1)
        if spp >= 2 * _BLOCK and mip is not None:
            return self._cols(mip[0], mip[1], _BLOCK, off, w)
        return self._cols(arr, arr, 1, off, w)

    def _spectrogram(self, w, h):
        key = (self.vs, self.vl, w, h, id(self.orig), theme.current_name())
        if self._spec_cache[0] == key:
            return self._spec_cache[1]
        nfft = 2048
        pos = np.linspace(self.vs, self.vs + self.vl, w, endpoint=False).astype(np.int64)
        idx = np.clip(pos[:, None] + np.arange(nfft)[None, :] - nfft // 2, 0, self.n - 1)
        frames = self.orig[idx] * np.hanning(nfft)[None, :].astype(np.float32)
        db = 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) / nfft * 4 + 1e-9)
        freqs = np.geomspace(60, 16000, h)[::-1]
        bins = np.clip((freqs / (SR / 2) * (nfft // 2)).astype(int), 0, nfft // 2)
        img = np.clip((db[:, bins].T + 100) / 70, 0, 1)
        rgb = spec_lut()[(img * 255).astype(np.uint8)]
        argb = np.ascontiguousarray(
            np.dstack([rgb[..., 2], rgb[..., 1], rgb[..., 0], np.full(img.shape, 255, np.uint8)]))
        qimg = QImage(argb.data, w, h, w * 4, QImage.Format_ARGB32).copy()
        self._spec_cache = (key, qimg)
        return qimg

    def paintEvent(self, _):
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, theme.color("panel2"))
        if self.orig is None:
            p.setPen(theme.color("muted"))
            p.setFont(QFont("Segoe UI", 12))
            p.drawText(self.rect(), Qt.AlignCenter, "Drop an audio or video file here, or click  Open")
            return
        top = 18
        area = H - top
        if self.mode == "spec":
            p.drawImage(0, top, self._spectrogram(W, area))
        else:
            mid = top + area / 2
            lo, hi = self._wave(self.orig, self._mo, 0, W)
            lc = hc = None
            if self.clean is not None:
                lc, hc = self._wave(self.clean, self._mc, self.clean_off, W)
            scale = 1.0
            if self.autoscale:
                pk = np.nanmax(np.abs(np.r_[lo, hi, lc if lc is not None else [0], hc if hc is not None else [0]]))
                scale = 0.95 / max(pk, 0.02) if np.isfinite(pk) else 1.0
                scale = min(scale, 40.0)
            amp = area / 2 * scale
            p.setPen(QPen(theme.color("grid"), 1))
            p.drawLine(0, int(mid), W, int(mid))
            for arrs, color in (((lo, hi), theme.color("wave")), ((lc, hc), theme.color("accent"))):
                if arrs[0] is None:
                    continue
                p.setPen(QPen(color, 1))
                for x in range(W):
                    if np.isnan(arrs[0][x]):
                        continue
                    y1 = mid - float(np.clip(arrs[1][x] * amp, -area / 2, area / 2))
                    y2 = mid - float(np.clip(arrs[0][x] * amp, -area / 2, area / 2))
                    p.drawLine(x, int(y1), x, int(max(y2, y1 + 1)))
        # selection
        if self.sel:
            x0, x1 = self.sample_to_x(self.sel[0]), self.sample_to_x(self.sel[1])
            p.fillRect(QRectF(x0, top, max(x1 - x0, 1), H - top), theme.color("accent", 60))
            p.setPen(QPen(theme.color("accent"), 1))
            p.drawLine(int(x0), top, int(x0), H)
            p.drawLine(int(x1), top, int(x1), H)
        # ruler
        p.fillRect(0, 0, W, top, theme.color("panel"))
        p.setPen(theme.color("muted"))
        p.setFont(QFont("Segoe UI", 8))
        dur = self.vl / SR
        step = next((s for s in (0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600) if dur / s <= 12), 1200)
        t = int(self.vs / SR / step) * step
        while t <= (self.vs + self.vl) / SR:
            x = self.sample_to_x(t * SR)
            if 0 <= x <= W:
                p.drawLine(int(x), 10, int(x), top)
                m, s = divmod(t, 60)
                p.drawText(QPointF(x + 3, 11), f"{int(m)}:{s:04.1f}" if step < 1 else f"{int(m)}:{int(s):02d}")
            t += step
        # playhead
        x = self.sample_to_x(self.playhead)
        if 0 <= x <= W:
            p.setPen(QPen(theme.color("text"), 2))
            p.drawLine(int(x), 0, int(x), H)


class WaveformPanel(QWidget):
    """WaveformView + horizontal scrollbar."""

    def __init__(self):
        super().__init__()
        self.view = WaveformView()
        self.bar = QScrollBar(Qt.Horizontal)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.bar)
        self.view.viewChanged.connect(self._sync)
        self.bar.valueChanged.connect(self._scrolled)
        self._busy = False

    def _sync(self):
        self._busy = True
        v = self.view
        self.bar.setRange(0, max(v.n - v.vl, 0))
        self.bar.setPageStep(v.vl)
        self.bar.setValue(v.vs)
        self._busy = False

    def _scrolled(self, val):
        if not self._busy:
            self.view.vs = val
            self.view.update()
