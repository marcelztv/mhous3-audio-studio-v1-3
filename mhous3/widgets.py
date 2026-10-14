"""Reusable interface pieces shared by all three workspaces."""
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
                               QMessageBox, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget)

from . import settings as cfg, theme
from .store import Store  # noqa: F401  (re-exported)
from .waveform import spec_lut


def tip(widget, text):
    widget.setToolTip(text)
    widget.setStatusTip(text)
    return widget


# ======================================================================== bound controls
class SliderRow(QWidget):
    def __init__(self, store, key, lo, hi, step=1.0, dec=0, suffix="", tip_text=""):
        super().__init__()
        self.store, self.key, self.step, self.dec, self.suffix = store, key, step, dec, suffix
        self.is_float = isinstance(cfg.DEFAULTS[key], float)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.sl = QSlider(Qt.Horizontal)
        self.sl.setRange(int(round(lo / step)), int(round(hi / step)))
        self.lab = QLabel()
        self.lab.setMinimumWidth(70)
        self.lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.sl, 1)
        lay.addWidget(self.lab)
        self.sl.valueChanged.connect(self._moved)
        store.changed.connect(self._refresh)
        self._refresh("*")
        if tip_text:
            tip(self, tip_text)

    def _moved(self, v):
        val = round(v * self.step, 4)
        self.store.set(self.key, float(val) if self.is_float else int(round(val)))
        self._label()

    def _label(self):
        self.lab.setText(f"{self.sl.value() * self.step:.{self.dec}f}{self.suffix}")

    def _refresh(self, key):
        if key in ("*", self.key):
            self.sl.blockSignals(True)
            self.sl.setValue(int(round(self.store.s[self.key] / self.step)))
            self.sl.blockSignals(False)
            self._label()


class ComboRow(QComboBox):
    def __init__(self, store, key, items, tip_text=""):
        super().__init__()
        self.store, self.key = store, key
        self.addItems(items)
        self.currentTextChanged.connect(lambda t: self.store.set(self.key, t))
        store.changed.connect(self._refresh)
        self._refresh("*")
        if tip_text:
            tip(self, tip_text)

    def _refresh(self, key):
        if key in ("*", self.key):
            v = str(self.store.s[self.key])
            if self.findText(v) < 0:
                self.addItem(v)
            self.blockSignals(True)
            self.setCurrentText(v)
            self.blockSignals(False)


class NumComboRow(QComboBox):
    """Combo box whose choices map to numbers (sample rate, bit depth)."""

    def __init__(self, store, key, choices, tip_text=""):
        super().__init__()
        self.store, self.key, self.choices = store, key, choices          # choices: {label: value}
        self.addItems(list(choices))
        self.currentTextChanged.connect(lambda t: self.store.set(self.key, self.choices[t]))
        store.changed.connect(self._refresh)
        self._refresh("*")
        if tip_text:
            tip(self, tip_text)

    def _refresh(self, key):
        if key in ("*", self.key):
            lab = next((k for k, v in self.choices.items() if v == self.store.s[self.key]), None)
            if lab:
                self.blockSignals(True)
                self.setCurrentText(lab)
                self.blockSignals(False)


class FormatCombo(ComboRow):
    def __init__(self, store):
        super().__init__(store, "out_format", list(cfg.FORMATS), "File type to save. The list adapts to the channels you chose.")
        store.changed.connect(self._channels)
        self._channels("*")

    def _channels(self, key):
        if key not in ("*", "out_channels"):
            return
        ok = cfg.formats_for(self.store.s["out_channels"])
        cur = self.store.s["out_format"]
        self.blockSignals(True)
        self.clear()
        self.addItems(ok)
        if cur not in ok:
            cur = "Dolby Digital AC-3" if self.store.s["out_channels"] == "Dolby Digital 5.1" else ok[0]
            self.store.s["out_format"] = cur
        self.setCurrentText(cur)
        self.blockSignals(False)


class CheckRow(QCheckBox):
    def __init__(self, store, key, text, tip_text=""):
        super().__init__(text)
        self.store, self.key = store, key
        self.toggled.connect(lambda v: self.store.set(self.key, bool(v)))
        store.changed.connect(self._refresh)
        self._refresh("*")
        tip(self, tip_text or text)

    def _refresh(self, key):
        if key in ("*", self.key):
            self.blockSignals(True)
            self.setChecked(bool(self.store.s[self.key]))
            self.blockSignals(False)


def group(title, rows):
    box = QGroupBox(title)
    f = QFormLayout(box)
    f.setLabelAlignment(Qt.AlignLeft)
    f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    for r in rows:
        if isinstance(r, tuple):
            if not r[1].toolTip():
                tip(r[1], f"{r[0].strip()}: change this to adjust the sound. Hover over a control for help.")
            f.addRow(r[0], r[1])
        else:
            f.addRow(r)
    return box


def hint(text):
    l = QLabel(text)
    l.setObjectName("hint")
    l.setWordWrap(True)
    return l


def scrolled(widget):
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(widget)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return sa


# ======================================================================== knob
class Knob(QWidget):
    """Large rotary macro control. Drag up/down, scroll, or double-click to reset."""

    def __init__(self, store, key, label, lo, hi, step=1.0, unit="", dec=0, default=None, tip_text=""):
        super().__init__()
        self.store, self.key, self.label = store, key, label
        self.lo, self.hi, self.step, self.unit, self.dec = lo, hi, step, unit, dec
        self.is_float = isinstance(cfg.DEFAULTS[key], float)
        self.default = cfg.DEFAULTS[key] if default is None else default
        self.setMinimumSize(118, 168)
        self.setMouseTracking(True)
        self._hover = False
        self._drag = None
        store.changed.connect(lambda k: self.update() if k in ("*", key) else None)
        tip(self, tip_text or label)

    def value(self):
        return float(self.store.s[self.key])

    def _set(self, v):
        v = min(self.hi, max(self.lo, round(v / self.step) * self.step))
        self.store.set(self.key, float(round(v, 4)) if self.is_float else int(round(v)))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pal = theme.palette()
        w, h = self.width(), self.height()
        d = min(w - 16, h - 54)
        rect = QRectF((w - d) / 2, 6, d, d)
        frac = (self.value() - self.lo) / (self.hi - self.lo)
        track = QPen(theme.color("border"), 9, Qt.SolidLine, Qt.RoundCap)
        p.setPen(track)
        p.drawArc(rect.adjusted(5, 5, -5, -5), 225 * 16, -270 * 16)
        p.setPen(QPen(theme.color("accent_hi" if self._hover else "accent"), 9, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect.adjusted(5, 5, -5, -5), 225 * 16, int(-270 * 16 * frac))
        inner = rect.adjusted(d * 0.2, d * 0.2, -d * 0.2, -d * 0.2)
        g = QLinearGradient(inner.topLeft(), inner.bottomRight())
        g.setColorAt(0, QColor(pal["panel2"]))
        g.setColorAt(1, QColor(pal["card"]))
        p.setBrush(g)
        p.setPen(QPen(theme.color("border"), 1))
        p.drawEllipse(inner)
        ang = math.radians(225 - 270 * frac)
        c = inner.center()
        r = inner.width() / 2
        p.setPen(QPen(theme.color("text"), 3, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(c.x() + math.cos(ang) * r * 0.35, c.y() - math.sin(ang) * r * 0.35),
                   QPointF(c.x() + math.cos(ang) * r * 0.85, c.y() - math.sin(ang) * r * 0.85))
        f = QFont(self.font())
        f.setPixelSize(15)
        f.setBold(True)
        p.setFont(f)
        p.setPen(theme.color("text"))
        p.drawText(QRectF(0, d + 8, w, 20), Qt.AlignCenter, f"{self.value():+.{self.dec}f}{self.unit}" if self.lo < 0 else
                   f"{self.value():.{self.dec}f}{self.unit}")
        f.setPixelSize(12)
        f.setBold(False)
        p.setFont(f)
        p.setPen(theme.color("muted"))
        p.drawText(QRectF(0, d + 28, w, 18), Qt.AlignCenter, self.label)

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = (e.position().y(), self.value())

    def mouseMoveEvent(self, e):
        if self._drag:
            fine = 0.25 if e.modifiers() & Qt.ShiftModifier else 1.0
            self._set(self._drag[1] + (self._drag[0] - e.position().y()) / 160 * (self.hi - self.lo) * fine)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        self._set(self.default)

    def wheelEvent(self, e):
        self._set(self.value() + (1 if e.angleDelta().y() > 0 else -1) * self.step * (5 if self.hi - self.lo > 40 else 2))


# ======================================================================== preset cards
class PresetCard(QFrame):
    clicked = Signal(str)

    def __init__(self, name, desc, special=False):
        super().__init__()
        self.name = name
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("selected", False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(3)
        t = QLabel(("* " if special else "") + name)
        t.setObjectName("cardtitle")
        t.setWordWrap(True)
        d = QLabel(desc)
        d.setObjectName("carddesc")
        d.setWordWrap(True)
        lay.addWidget(t)
        lay.addWidget(d)
        lay.addStretch(1)
        tip(self, f"{name}: {desc} Click to apply.")

    def set_selected(self, v):
        self.setProperty("selected", v)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.name)


class PresetCards(QWidget):
    """Visible gallery of presets. Horizontal strip (FastTrack) or vertical list (dock)."""
    picked = Signal(str)
    AUTO = "Auto Clean (best for this audio)"

    def __init__(self, user_presets=None, horizontal=True, include_auto=True):
        super().__init__()
        self.cards = {}
        self.horizontal = horizontal
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QWidget()
        self.lay = QHBoxLayout(inner) if horizontal else QVBoxLayout(inner)
        self.lay.setContentsMargins(2, 2, 2, 2)
        self.lay.setSpacing(8)
        sa = scrolled(inner)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded if horizontal else Qt.ScrollBarAlwaysOff)
        sa.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff if horizontal else Qt.ScrollBarAsNeeded)
        outer.addWidget(sa)
        if horizontal:
            sa.setFixedHeight(112)
        if include_auto:
            self._add(self.AUTO, "We listen to your file and choose the settings for you.", True)
        for n in cfg.PRESETS:
            self._add(n, cfg.PRESET_DESC.get(n, ""))
        for n in (user_presets or {}):
            self._add("* " + n, "Your own saved preset.")
        self.lay.addStretch(1)

    def _add(self, name, desc, special=False):
        c = PresetCard(name, desc, special)
        if self.horizontal:
            c.setFixedWidth(190)
        c.clicked.connect(self.picked.emit)
        self.cards[name] = c
        self.lay.addWidget(c)

    def select(self, name):
        for n, c in self.cards.items():
            c.set_selected(n == name)


# ======================================================================== live spectrogram
class LiveSpectrogram(QWidget):
    """Scrolling, flowing spectrogram that follows playback (shows what you are hearing: A or B)."""
    COLS, ROWS, HOP, NFFT = 720, 150, 960, 2048

    def __init__(self, session):
        super().__init__()
        self.session = session
        self.setMinimumHeight(110)
        self.img = np.zeros((self.ROWS, self.COLS), np.uint8)
        self.last = None
        self.last_src = None
        self._qimg = None
        self._theme = None
        self._dirty = True
        self.overview = True
        self.win = np.hanning(self.NFFT).astype(np.float32)
        freqs = np.geomspace(60, 16000, self.ROWS)[::-1]
        self.bins = np.clip((freqs / 24000 * (self.NFFT // 2)).astype(int), 0, self.NFFT // 2)
        session.tick.connect(self.advance)
        session.audioChanged.connect(self.reset)
        session.previewReady.connect(self.reset)

    def _cols(self, positions):
        arr, off = self.session.spec_source()
        if arr is None:
            return np.zeros((self.ROWS, len(positions)), np.uint8)
        idx = positions[:, None] - off + np.arange(self.NFFT)[None, :] - self.NFFT // 2
        bad = (idx < 0) | (idx >= len(arr))
        fr = arr[np.clip(idx, 0, len(arr) - 1)] * self.win[None, :]
        fr[bad] = 0
        db = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) / self.NFFT * 4 + 1e-9)
        return (np.clip((db[:, self.bins] + 100) / 70, 0, 1).T * 255).astype(np.uint8)

    AHEAD = 0.25    # fraction of the window shown *ahead* of the playhead while scrolling

    def reset(self):
        self.last = None
        self._prime()

    def _prime(self):
        s = self.session
        if s.audio is None:
            return
        pos = s.global_pos()
        if not s.player.playing and pos == 0:                       # idle at the start: overview of the whole file
            self.overview = True
            self.img = self._cols(np.linspace(0, max(s.n - 1, 1), self.COLS).astype(np.int64))
        else:                                                         # scrolling window: past + a little future
            self.overview = False
            ahead = int(self.COLS * self.AHEAD)
            self.img = self._cols(pos + (np.arange(self.COLS) - (self.COLS - 1 - ahead)) * self.HOP)
        self.last, self._dirty = pos, True
        self.update()

    def advance(self):
        s = self.session
        if s.audio is None or not self.isVisible():
            return
        if s.player.src != self.last_src:
            self.last_src = s.player.src
            self._prime()
            return
        pos = s.global_pos()
        if self.overview:
            if s.player.playing or pos != 0:
                self._prime()
            return
        if self.last is None or pos < self.last or pos - self.last > self.HOP * 80:
            self._prime()
            return
        k = (pos - self.last) // self.HOP
        if k >= 1:
            ahead = int(self.COLS * self.AHEAD)
            head = self.last + ahead * self.HOP
            cols = self._cols(head + self.HOP * np.arange(1, k + 1))
            self.img = np.concatenate([self.img[:, k:], cols], axis=1) if k < self.COLS else cols[:, -self.COLS:]
            self.last += k * self.HOP
            self._dirty = True
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, theme.color("panel2"))
        if self.session.audio is None:
            p.setPen(theme.color("muted"))
            p.drawText(self.rect(), Qt.AlignCenter, "The live spectrogram appears here when you open a file")
            return
        if self._dirty or self._theme != theme.current_name():
            rgb = spec_lut()[self.img]
            argb = np.ascontiguousarray(np.dstack([rgb[..., 2], rgb[..., 1], rgb[..., 0],
                                                   np.full(self.img.shape, 255, np.uint8)]))
            self._qimg = QImage(argb.data, self.COLS, self.ROWS, self.COLS * 4, QImage.Format_ARGB32).copy()
            self._dirty, self._theme = False, theme.current_name()
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(QRectF(0, 0, W, H), self._qimg)
        nx = (self.session.global_pos() / max(self.session.n, 1)) * W if self.overview else W * (1 - self.AHEAD)
        p.setPen(QPen(theme.color("accent"), 2))
        p.drawLine(int(nx), 0, int(nx), H)
        f = QFont(self.font())
        f.setPixelSize(10)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255, 170) if theme.current_name() == "dark" else QColor(20, 40, 80, 190))
        for hz, lab in ((100, "100 Hz"), (1000, "1 kHz"), (10000, "10 kHz")):
            y = H - (math.log(hz / 60) / math.log(16000 / 60)) * H
            p.drawText(4, int(y) - 2, lab)


# ======================================================================== panels
class PresetBar(QWidget):
    """Compact preset menu (used in the Basic dock) - applies through `on_pick` so popups and Auto Clean work."""

    def __init__(self, store, user_presets, on_pick):
        super().__init__()
        self.store, self.user, self.on_pick = store, user_presets, on_pick
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        self.combo.setMinimumContentsLength(18)
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        tip(self.combo, "Choose a ready-made group of settings.")
        self.fill()
        self.combo.setPlaceholderText("Choose a preset...")
        self.combo.setCurrentIndex(-1)
        self.combo.activated.connect(lambda i: self.on_pick(self.combo.itemText(i)))
        save = tip(QPushButton("Save..."), "Save the current settings as your own preset.")
        save.clicked.connect(self._save)
        dele = tip(QPushButton("Delete"), "Delete one of your own saved presets.")
        dele.clicked.connect(self._delete)
        lay.addWidget(QLabel("Preset"))
        lay.addWidget(self.combo, 1)
        lay.addWidget(save)
        lay.addWidget(dele)

    def names(self):
        return list(cfg.PRESETS) + ["* " + n for n in self.user]

    def fill(self, select=None):
        self.combo.clear()
        self.combo.addItems(self.names())
        if select:
            self.combo.setCurrentText(select)
        else:
            self.combo.setCurrentIndex(-1)

    def _save(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if ok and name.strip():
            self.user[name.strip()] = dict(self.store.s)
            cfg.save_user_presets(self.user)
            self.fill("* " + name.strip())
            self.store.changed.emit("__presets__")

    def _delete(self):
        t = self.combo.currentText()
        if t.startswith("* ") and t[2:] in self.user:
            del self.user[t[2:]]
            cfg.save_user_presets(self.user)
            self.fill()
            self.store.changed.emit("__presets__")
        else:
            QMessageBox.information(self, "Presets", "Built-in presets can't be deleted. Only your own presets (marked *) can.")


class OutputPanel(QGroupBox):
    def __init__(self, store):
        super().__init__("Output")
        f = QFormLayout(self)
        f.addRow("Channels", ComboRow(store, "out_channels", cfg.OUT_CHANNELS,
                                      "Mono, Stereo, Dolby Surround (Lt/Rt) or Dolby Digital 5.1."))
        f.addRow("Format", FormatCombo(store))
        f.addRow("Sample rate", NumComboRow(store, "out_sr", {f"{r / 1000:g} kHz": r for r in cfg.SAMPLE_RATES}, "Sample rate of the saved file (44.1 to 192 kHz). Processing always happens at 48 kHz."))
        f.addRow("Bit depth", NumComboRow(store, "out_bits", cfg.BIT_DEPTHS, "16-bit adds dither automatically. MP3/AAC/AC-3 ignore bit depth."))
        f.addRow(CheckRow(store, "loudness_on", "Match loudness", "Bring the whole file to the loudness target below."))
        f.addRow("Target", ComboRow(store, "loudness_target", list(cfg.LOUDNESS_TARGETS), "How loud the finished file should be."))
        f.addRow("Custom", SliderRow(store, "lufs_custom", -40, -9, 0.5, 1, " LUFS", "Used when Target is Custom."))
        f.addRow("Peak ceiling", SliderRow(store, "ceiling", -6, 0, 0.1, 1, " dB", "Nothing in the output goes above this level."))


def basic_panel(store, on_pick, user_presets):
    """Plain-language controls for people with no sound-design knowledge."""
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.addWidget(PresetBar(store, user_presets, on_pick))
    lay.addWidget(group("Make it louder and clearer", [
        ("Make voice louder", SliderRow(store, "volume_boost", -12, 24, 0.5, 1, " dB", "Raises the overall volume. Peaks are protected so it will not distort.")),
        ("Make voice clearer", SliderRow(store, "clarity", 0, 100, 1, 0, " %", "Makes words easier to understand.")),
        ("Make voice fuller", SliderRow(store, "warmth", 0, 100, 1, 0, " %", "Adds depth and body for a richer voice.")),
        ("Professional polish", SliderRow(store, "polish", 0, 100, 1, 0, " %", "Adds the smooth, finished sound of professional recordings.")),
    ]))
    lay.addWidget(group("Remove problems", [
        ("Remove background noise", SliderRow(store, "nr_amount", 0, 100, 1, 0, " %", "Reduces hiss, hum, fans and street noise.")),
        ("Remove echo", SliderRow(store, "derev_amount", 0, 100, 1, 0, " %", "Reduces the hollow echo of rooms.")),
        ("Fix distortion", ComboRow(store, "declip", ["Auto", "On", "Off"], "Repairs sound that was recorded too loud and crackles.")),
        ("Electrical buzz", ComboRow(store, "hum", ["Off", "50", "60"], "Removes mains hum. Use 50 in South Africa and Europe, 60 in the Americas.")),
    ]))
    lay.addWidget(group("Add space", [
        ("Room feel", SliderRow(store, "room_expansion", 0, 100, 1, 0, " %", "Gives a dry voice a natural sense of room.")),
        ("Reverb type", ComboRow(store, "reverb_type", cfg.reverb_choices(), "The kind of space for the reverb.")),
        ("Reverb amount", SliderRow(store, "reverb_mix", 0, 100, 1, 0, " %", "0 is completely dry.")),
    ]))
    lay.addStretch(1)
    return w


def advanced_panel(store):
    """Full parametric controls for skilled users and professionals."""
    w = QWidget()
    lay = QVBoxLayout(w)
    S = SliderRow
    lay.addWidget(group("Input and repair", [
        ("Source channels", ComboRow(store, "input_mode", ["Auto", "Mono", "Stereo"], "Auto treats identical L/R as mono.")),
        ("Declip max run", S(store, "declip_max_run", 8, 200, 4, 0, " smp", "Longest clipped stretch to rebuild.")),
        ("Low cut", S(store, "lowcut", 0, 300, 5, 0, " Hz", "Removes rumble below this frequency. 0 = off.")),
        ("High cut", S(store, "highcut", 0, 20000, 500, 0, " Hz", "Removes extreme highs above this. 0 = off.")),
    ]))
    lay.addWidget(group("Noise and echo engines", [
        ("Noise engine", ComboRow(store, "nr_engine", ["Auto", "AI (DeepFilterNet)", "Built-in"], "Auto uses the AI engine when available.")),
        ("Built-in sensitivity", S(store, "nr_sens", 1, 5, 0.1, 1, "x", "Higher removes more noise but can sound thin.")),
        ("Room decay (RT60)", S(store, "derev_rt60", 0.2, 3.0, 0.05, 2, " s", "About how long the room echo lasts.")),
    ]))
    names = ["Low shelf", "Low-mid", "Mid", "Presence", "Upper-mid", "High shelf"]
    rows = []
    for i in range(1, 7):
        rows += [(f"{i}. {names[i - 1]} gain", S(store, f"eq{i}_g", -15, 15, 0.5, 1, " dB", "Boost or cut this band.")),
                 ("    frequency", S(store, f"eq{i}_f", 20, 16000, 10, 0, " Hz", "Centre or corner frequency.")),
                 ("    Q", S(store, f"eq{i}_q", 0.2, 5, 0.1, 1, "", "Band width: higher is narrower."))]
    lay.addWidget(group("6-band parametric EQ", rows))
    lay.addWidget(group("Gate / expander", [
        CheckRow(store, "gate_on", "Enable", "Lowers very quiet sound between words."),
        ("Threshold", S(store, "gate_thr", -80, -20, 1, 0, " dB", "Sound below this is reduced.")),
        ("Ratio", S(store, "gate_ratio", 1, 6, 0.1, 1, ":1", "How strongly quiet sound is reduced.")),
        ("Max reduction", S(store, "gate_range", -40, -3, 1, 0, " dB", "Most the gate will turn anything down.")),
        ("Attack", S(store, "gate_attack", 1, 50, 1, 0, " ms", "How fast it opens.")),
        ("Release", S(store, "gate_release", 20, 800, 10, 0, " ms", "How slowly it closes.")),
    ]))
    lay.addWidget(group("De-esser", [
        CheckRow(store, "deess_on", "Enable", "Softens harsh S sounds."),
        ("Frequency", S(store, "deess_freq", 3500, 10000, 100, 0, " Hz", "Where the harsh S lives.")),
        ("Threshold", S(store, "deess_thr", -60, -15, 1, 0, " dB", "Lower = more de-essing.")),
        ("Ratio", S(store, "deess_ratio", 1, 10, 0.5, 1, ":1", "Strength of reduction.")),
    ]))
    lay.addWidget(group("Compressor", [
        CheckRow(store, "comp_on", "Enable", "Evens out loud and quiet speech."),
        ("Threshold", S(store, "comp_thr", -50, 0, 1, 0, " dB", "Sound above this is turned down.")),
        ("Ratio", S(store, "comp_ratio", 1, 10, 0.1, 1, ":1", "How much louder sound is reduced.")),
        ("Attack", S(store, "comp_attack", 1, 100, 1, 0, " ms", "Reaction speed.")),
        ("Release", S(store, "comp_release", 20, 800, 10, 0, " ms", "Recovery speed.")),
        ("Knee", S(store, "comp_knee", 0, 24, 1, 0, " dB", "Softness of the transition.")),
        ("Make-up gain", S(store, "comp_makeup", 0, 18, 0.5, 1, " dB", "Level added after compression.")),
    ]))
    lay.addWidget(group("Colour and width", [
        CheckRow(store, "sat_on", "Tape-style saturation", "Adds warm harmonic colour."),
        ("Drive", S(store, "sat_drive", 0.5, 6, 0.1, 1, "x", "Saturation intensity.")),
        ("Mix", S(store, "sat_mix", 0, 100, 1, 0, " %", "Blend of saturated and clean signal.")),
        ("Stereo width", S(store, "width", 0, 200, 1, 0, " %", "100% = unchanged. Ignored for mono.")),
    ]))
    lay.addWidget(group("Reverb details", [
        ("Room", ComboRow(store, "reverb_type", cfg.reverb_choices(), "Type of space.")),
        ("Amount", S(store, "reverb_mix", 0, 100, 1, 0, " %", "Reverb level.")),
        ("Pre-delay", S(store, "reverb_predelay", -1, 120, 1, 0, " ms", "-1 = the room's default.")),
        ("Damping", S(store, "reverb_damp", -1, 1, 0.05, 2, "", "-1 = default. Higher = darker tail.")),
        ("Decay (RT60)", S(store, "reverb_rt60", 0, 8, 0.1, 1, " s", "0 = the room's default.")),
        ("Reverb low cut", S(store, "reverb_lowcut", 20, 600, 10, 0, " Hz", "Keeps the reverb clean.")),
        ("Reverb high cut", S(store, "reverb_highcut", 2000, 16000, 250, 0, " Hz", "Softens the top of the reverb.")),
    ]))
    lay.addWidget(group("Dolby / 5.1 mix", [
        ("Dialogue to centre", S(store, "center_level", 0, 100, 1, 0, " %", "How much voice goes to the centre speaker.")),
        ("Surround level", S(store, "surround_level", 0, 100, 1, 0, " %", "Ambience sent to the rear speakers.")),
        ("LFE (subwoofer)", S(store, "lfe_level", 0, 100, 1, 0, " %", "Low frequencies sent to the sub.")),
    ]))
    lay.addStretch(1)
    return w
