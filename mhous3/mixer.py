"""Console-style mixer (strips, sends, inserts, buses, VCAs, master), effect editor and loudness meters."""
import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QMenu, QPlainTextEdit, QPushButton, QScrollArea, QSlider, QToolButton, QVBoxLayout, QWidget)

from . import dsp, fx, jsfx, theme
from .dsp import SR
from .session import Task
from .widgets import tip

SMALL = "QPushButton, QToolButton { padding: 2px 5px; min-width: 0px; }"


def _gone(w):
    """Remove a widget immediately (hide + detach) and delete it later - no ghost frames while rebuilding."""
    w.hide()
    w.setParent(None)
    w.deleteLater()


TRACK_AUTO_MODES = ["Read", "Off", "Write", "Latch", "Touch", "Touch/Latch"]


# ======================================================================== effect editor
class FxEditorDialog(QDialog):
    """Auto-generated controls for any effect; script editor for JSFX."""

    def __init__(self, project, tid, index, parent=None):
        super().__init__(parent)
        self.p, self.tid, self.index = project, tid, index
        self.slot = project._chain(tid)[index]
        kind = self.slot["kind"]
        self.setWindowTitle("JSFX script" if kind == "jsfx" else fx.EFFECTS[kind]["name"])
        self.setMinimumWidth(560)
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.byp = QCheckBox("Bypass")
        self.byp.setChecked(self.slot["bypass"])
        self.byp.toggled.connect(lambda b: project.set_insert(tid, index, bypass=bool(b)))
        tip(self.byp, "Turn this effect off without removing it.")
        top.addWidget(self.byp)
        top.addStretch(1)
        if kind in fx.EFFECTS:
            pre = [n for n, p in fx.PLUGIN_PRESETS.items() if p["kind"] == kind]
            if pre:
                top.addWidget(QLabel("Preset"))
                self.pre = QComboBox()
                self.pre.addItems(["-"] + pre)
                self.pre.activated.connect(self._preset)
                top.addWidget(self.pre)
        v.addLayout(top)
        self.body = QWidget()
        self.form = QFormLayout(self.body)
        v.addWidget(self.body)
        if kind == "jsfx":
            self.ex = QComboBox()
            self.ex.addItems(["Load example..."] + list(jsfx.EXAMPLES))
            self.ex.activated.connect(self._example)
            v.addWidget(self.ex)
            self.edit = QPlainTextEdit(self.slot["data"].get("script", ""))
            self.edit.setFont(QFont("Consolas", 10))
            self.edit.setMinimumHeight(220)
            v.addWidget(self.edit)
            self.err = QLabel("")
            self.err.setWordWrap(True)
            v.addWidget(self.err)
            apply = QPushButton("Apply script")
            apply.setObjectName("primary")
            apply.clicked.connect(self._apply_script)
            tip(apply, "Compile the script and build its sliders.")
            v.addWidget(apply)
        row = QHBoxLayout()
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        v.addLayout(row)
        self._build()

    def _clear(self):
        while self.form.rowCount():
            self.form.removeRow(0)

    def _build(self):
        self._clear()
        kind = self.slot["kind"]
        if kind == "jsfx":
            try:
                info = jsfx.parse_header(self.slot["data"].get("script", ""))
            except Exception:
                return
            for s in info["sliders"]:
                key = f"slider{s['idx']}"
                val = self.slot["params"].get(key, s["default"])
                if s["options"]:
                    cb = QComboBox()
                    cb.addItems(s["options"])
                    cb.setCurrentIndex(int(val))
                    cb.activated.connect(lambda i, key=key: self.p.set_insert(self.tid, self.index, params={key: float(i)}))
                    self.form.addRow(s["label"], cb)
                else:
                    self.form.addRow(s["label"], self._slider(key, s["min"], s["max"], s["step"], val, ""))
            return
        for prm in fx.EFFECTS[kind]["params"]:
            key = prm[0]
            if isinstance(prm[2], list):
                cb = QComboBox()
                cb.addItems(prm[2])
                cb.setCurrentText(str(self.slot["params"].get(key, prm[3])))
                cb.activated.connect(lambda i, key=key, cb=cb: self.p.set_insert(self.tid, self.index, params={key: cb.itemText(i)}))
                self.form.addRow(prm[1], cb)
            else:
                self.form.addRow(prm[1], self._slider(key, prm[2], prm[3], prm[5], self.slot["params"].get(key, prm[4]), prm[6]))

    def _slider(self, key, lo, hi, step, val, unit):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        sl = QSlider(Qt.Horizontal)
        n = max(1, int(round((hi - lo) / step)))
        sl.setRange(0, n)
        sl.setValue(int(round((float(val) - lo) / step)))
        lab = QLabel()
        lab.setMinimumWidth(80)
        lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        dec = 0 if step >= 1 else (1 if step >= 0.1 else 2)

        def moved(k):
            v = round(lo + k * step, 4)
            lab.setText(f"{v:.{dec}f} {unit}")
            self.p.set_insert(self.tid, self.index, params={key: v})
        sl.valueChanged.connect(moved)
        lab.setText(f"{float(val):.{dec}f} {unit}")
        h.addWidget(sl, 1)
        h.addWidget(lab)
        return w

    def _preset(self, i):
        name = self.pre.itemText(i)
        if name in fx.PLUGIN_PRESETS:
            self.p.set_insert(self.tid, self.index, params=dict(fx.PLUGIN_PRESETS[name]["params"]))
            self.slot = self.p._chain(self.tid)[self.index]
            self._build()

    def _example(self, i):
        if i > 0:
            self.edit.setPlainText(jsfx.EXAMPLES[self.ex.itemText(i)])
            self._apply_script()

    def _apply_script(self):
        text = self.edit.toPlainText()
        try:
            jsfx.compile_script(text)
        except jsfx.JsfxError as e:
            self.err.setStyleSheet(f"color: {theme.palette()['danger']};")
            self.err.setText(f"Script problem: {e}")
            return
        self.err.setStyleSheet(f"color: {theme.palette()['good']};")
        self.err.setText("Script compiled.")
        self.p.set_insert(self.tid, self.index, data={"script": text}, params={})
        self.slot = self.p._chain(self.tid)[self.index]
        self._build()


def insert_menu(project, tid, parent, on_added=None):
    """Menu to add an effect, a JSFX script or a plugin preset to a chain (tid 0 = master)."""
    m = QMenu(parent)
    for cat, effects in fx.categories().items():
        sub = m.addMenu(cat)
        for e in effects:
            sub.addAction(e["name"], lambda eid=e["id"]: (project.add_insert(tid, eid), on_added and on_added()))
    js = m.addMenu("JSFX (script)")
    js.addAction("Empty script...", lambda: (project.add_insert(tid, "jsfx", {}, {"script": "desc:My effect\nslider1:0<-24,24,0.5>Gain (dB)\n@slider\ng = 10^(slider1/20);\n@sample\nspl0 *= g;\nspl1 *= g;\n"}), on_added and on_added()))
    for name, src in jsfx.EXAMPLES.items():
        js.addAction(name, lambda src=src: (project.add_insert(tid, "jsfx", {}, {"script": src}), on_added and on_added()))
    pm = m.addMenu("Plugin presets")
    for name, p in fx.PLUGIN_PRESETS.items():
        pm.addAction(name, lambda p=p: (project.add_insert(tid, p["kind"], dict(p["params"])), on_added and on_added()))
    return m


# ======================================================================== mixer
class Strip(QFrame):
    def __init__(self, mixer, kind, obj):
        super().__init__()
        self.m, self.kind, self.obj = mixer, kind, obj          # kind: track / bus / vca / master
        self.p = mixer.p
        self.setObjectName("card")
        self.setFixedWidth(216)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(5)
        color = "#5cb6ff" if kind == "master" else (obj["color"] if kind == "vca" and "color" in obj else getattr(obj, "color", "#c792ea"))
        bar = QFrame()
        bar.setFixedHeight(4)
        bar.setStyleSheet(f"background: {color}; border-radius: 2px;")
        v.addWidget(bar)
        self.name = QLineEdit("Master" if kind == "master" else (obj["name"] if kind == "vca" else obj.name))
        self.name.setReadOnly(kind == "master")
        self.name.setStyleSheet("font-weight: 700; background: transparent; border: none;")
        self.name.editingFinished.connect(self._rename)
        tip(self.name, "Click to rename.")
        v.addWidget(self.name)
        if kind in ("track", "bus"):
            self.out = QComboBox()
            tip(self.out, "Where this track's signal goes: the master or a bus (submix).")
            self.out.activated.connect(self._route)
            self.vca = QComboBox()
            tip(self.vca, "Assign to a VCA master: it scales the fader without touching the audio path.")
            self.vca.activated.connect(self._set_vca)
            self.mode = QComboBox()
            self.mode.addItems(TRACK_AUTO_MODES)
            tip(self.mode, "Automation mode. Read plays your automation lanes; Off ignores them. Write/Latch/Touch recording needs the live engine (planned).")
            self.mode.activated.connect(lambda i: self.p.set_track(obj.id, auto_mode=TRACK_AUTO_MODES[i]))
            f = QFormLayout()
            f.setContentsMargins(0, 0, 0, 0)
            f.addRow("Out", self.out)
            f.addRow("VCA", self.vca)
            f.addRow("Auto", self.mode)
            v.addLayout(f)
        if kind in ("track", "bus", "master"):
            self.ins = QListWidget()
            self.ins.setFixedHeight(92)
            tip(self.ins, "Insert effects in order, top to bottom. Double-click to edit.")
            self.ins.itemDoubleClicked.connect(lambda _i: self._edit())
            v.addWidget(self.ins)
            r = QHBoxLayout()
            add = QToolButton()
            add.setText("+FX")
            add.setStyleSheet(SMALL)
            add.setPopupMode(QToolButton.InstantPopup)
            add.setMenu(insert_menu(self.p, self._chain_id(), self))
            tip(add, "Add an effect, JSFX script or plugin preset.")
            r.addWidget(add)
            for txt, fn, tt in (("Edit", self._edit, "Open the effect's controls."), ("Byp", self._bypass, "Bypass / enable the selected effect."),
                                ("^", lambda: self._move(-1), "Move up."), ("v", lambda: self._move(1), "Move down."), ("x", self._remove, "Remove.")):
                b = QPushButton(txt)
                b.setStyleSheet(SMALL)
                b.setFixedHeight(24)
                b.clicked.connect(lambda _=False, fn=fn: fn())
                tip(b, tt)
                r.addWidget(b)
            v.addLayout(r)
            if kind != "master":
                pr = QToolButton()
                pr.setText("Track preset  v")
                pr.setPopupMode(QToolButton.InstantPopup)
                menu = QMenu(pr)
                for name in fx.TRACK_PRESETS:
                    menu.addAction(name, lambda name=name: self.p.load_track_preset(self.obj.id, name))
                pr.setMenu(menu)
                tip(pr, "Replace this track's inserts with a ready-made chain.")
                v.addWidget(pr)
        if kind == "track":
            self.send_box = QVBoxLayout()
            v.addLayout(self.send_box)
            self.add_send = QToolButton()
            self.add_send.setText("+ Send  v")
            self.add_send.setPopupMode(QToolButton.InstantPopup)
            tip(self.add_send, "Send some of this track to a bus (for reverb, delay or a headphone mix).")
            v.addWidget(self.add_send)
        # pan + fader + buttons
        if kind in ("track", "bus"):
            self.pan = QSlider(Qt.Horizontal)
            self.pan.setRange(-100, 100)
            tip(self.pan, "Pan / balance. Double-click to centre.")
            self.pan.valueChanged.connect(lambda val: self.p.set_track(obj.id, pan=val / 100.0))
            self.pan.mouseDoubleClickEvent = lambda e: self.pan.setValue(0)
            v.addWidget(self.pan)
        h = QHBoxLayout()
        self.fader = QSlider(Qt.Vertical)
        self.fader.setRange(-600, 120)
        self.fader.setMinimumHeight(150)
        tip(self.fader, "Fader (dB). Double-click to reset to 0 dB.")
        self.fader.mouseDoubleClickEvent = lambda e: self.fader.setValue(0)
        self.fader.valueChanged.connect(self._fader)
        h.addStretch(1)
        h.addWidget(self.fader)
        col = QVBoxLayout()
        self.db = QLabel("0.0 dB")
        col.addWidget(self.db)
        self.mute = QPushButton("M")
        self.mute.setCheckable(True)
        tip(self.mute, "Mute.")
        self.mute.toggled.connect(self._mute)
        col.addWidget(self.mute)
        if kind == "track":
            self.solo = QPushButton("S")
            self.solo.setCheckable(True)
            tip(self.solo, "Solo.")
            self.solo.toggled.connect(lambda b: self.p.set_track(obj.id, solo=bool(b)))
            col.addWidget(self.solo)
        if kind == "master":
            self.level = QSlider(Qt.Vertical)
            self.level.setRange(0, 600)
            self.level.setEnabled(False)
            tip(self.level, "Output level while playing.")
            h.addWidget(self.level)
        col.addStretch(1)
        h.addLayout(col)
        h.addStretch(1)
        v.addLayout(h)
        self.sync()

    # ------------------------------------------------------------------ plumbing
    def _chain_id(self):
        return 0 if self.kind == "master" else self.obj.id

    def _chain(self):
        return self.p._chain(self._chain_id())

    def _rename(self):
        t = self.name.text().strip()
        if not t:
            return
        if self.kind == "vca":
            self.obj["name"] = t
            self.p.changed()
        elif self.kind != "master":
            self.p.set_track(self.obj.id, name=t)

    def _route(self, i):
        self.p.set_route(self.obj.id, self.out.itemData(i) or 0)

    def _set_vca(self, i):
        self.p.set_track(self.obj.id, vca=self.vca.itemData(i) or 0)

    def _sel(self):
        return self.ins.currentRow()

    def _edit(self):
        i = self._sel()
        if 0 <= i < len(self._chain()):
            FxEditorDialog(self.p, self._chain_id(), i, self).exec()
            self.m.refresh(force=True)

    def _bypass(self):
        i = self._sel()
        if 0 <= i < len(self._chain()):
            self.p.set_insert(self._chain_id(), i, bypass=not self._chain()[i]["bypass"])

    def _move(self, d):
        i = self._sel()
        self.p.move_insert(self._chain_id(), i, d)
        self.m.refresh(force=True)
        self.m.strip_for(self.kind, self.obj).ins.setCurrentRow(max(0, min(i + d, self.ins.count() - 1)))

    def _remove(self):
        self.p.remove_insert(self._chain_id(), self._sel())
        self.m.refresh(force=True)

    def _fader(self, val):
        db = val / 10.0
        self.db.setText(f"{db:+.1f} dB")
        if self.kind == "master":
            self.p.master["gain_db"] = db
            self.p.changed()
        elif self.kind == "vca":
            self.obj["gain_db"] = db
            self.p.changed()
        else:
            self.p.set_track(self.obj.id, gain_db=db)

    def _mute(self, b):
        if self.kind == "master":
            self.p.master["mute"] = bool(b)
            self.p.changed()
        elif self.kind == "vca":
            self.obj["mute"] = bool(b)
            self.p.changed()
        else:
            self.p.set_track(self.obj.id, mute=bool(b))

    def sync(self):
        """Refresh widget values from the project without triggering edits."""
        o, k = self.obj, self.kind
        gain = self.p.master["gain_db"] if k == "master" else (o["gain_db"] if k == "vca" else o.gain_db)
        mute = self.p.master["mute"] if k == "master" else (o["mute"] if k == "vca" else o.mute)
        for w, fn in ((self.fader, lambda: self.fader.setValue(int(round(gain * 10)))), (self.mute, lambda: self.mute.setChecked(bool(mute)))):
            w.blockSignals(True)
            fn()
            w.blockSignals(False)
        self.db.setText(f"{gain:+.1f} dB")
        if k in ("track", "bus"):
            self.pan.blockSignals(True)
            self.pan.setValue(int(round(o.pan * 100)))
            self.pan.blockSignals(False)
            self.mode.setCurrentText(o.auto_mode)
            self.out.blockSignals(True)
            self.out.clear()
            self.out.addItem("Master", 0)
            for b in self.p.tracks:
                if b.is_bus and b.id != o.id:
                    self.out.addItem(b.name, b.id)
            self.out.setCurrentIndex(max(0, self.out.findData(o.output)))
            self.out.blockSignals(False)
            self.vca.blockSignals(True)
            self.vca.clear()
            self.vca.addItem("None", 0)
            for v in self.p.vcas:
                self.vca.addItem(v["name"], v["id"])
            self.vca.setCurrentIndex(max(0, self.vca.findData(o.vca)))
            self.vca.blockSignals(False)
        if k == "track":
            self.solo.blockSignals(True)
            self.solo.setChecked(o.solo)
            self.solo.blockSignals(False)
        if k in ("track", "bus", "master"):
            row = self.ins.currentRow()
            self.ins.clear()
            for s in self._chain():
                nm = "JSFX: " + jsfx.parse_header(s["data"].get("script", ""))["desc"] if s["kind"] == "jsfx" else fx.EFFECTS[s["kind"]]["name"]
                self.ins.addItem(("(off) " if s["bypass"] else "") + nm)
            if 0 <= row < self.ins.count():
                self.ins.setCurrentRow(row)
        if k == "track":
            self._sends()

    def _sends(self):
        while self.send_box.count():
            it = self.send_box.takeAt(0)
            if it.widget():
                _gone(it.widget())
        t = self.obj
        for s in t.sends:
            bus = self.p.track(s["bus"])
            if not bus:
                continue
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel("> " + bus.name[:9]))
            sl = QSlider(Qt.Horizontal)
            sl.setRange(-600, 60)
            sl.setValue(int(s["db"] * 10))
            tip(sl, f"Send level to {bus.name} (dB).")
            sl.valueChanged.connect(lambda val, b=s["bus"], pre=s["pre"]: self.p.set_send(t.id, b, val / 10.0, self._pre(b)))
            h.addWidget(sl, 1)
            pre = QCheckBox("pre")
            pre.setChecked(bool(s.get("pre")))
            tip(pre, "Pre-fader: the send ignores this track's fader.")
            pre.toggled.connect(lambda c, b=s["bus"]: self.p.set_send(t.id, b, self._db(b), bool(c)))
            h.addWidget(pre)
            x = QPushButton("x")
            x.setStyleSheet(SMALL)
            x.setFixedSize(22, 22)
            x.clicked.connect(lambda _=False, b=s["bus"]: (self.p.remove_send(t.id, b), self.m.refresh(force=True)))
            h.addWidget(x)
            self.send_box.addWidget(w)
        menu = QMenu(self.add_send)
        used = {s["bus"] for s in t.sends}
        for b in self.p.tracks:
            if b.is_bus and b.id not in used:
                menu.addAction(b.name, lambda b=b: (self.p.set_send(t.id, b.id, -12.0, False), self.m.refresh(force=True)))
        if not menu.actions():
            menu.addAction("(add a bus first)").setEnabled(False)
        self.add_send.setMenu(menu)

    def _pre(self, bus):
        return next((bool(s.get("pre")) for s in self.obj.sends if s["bus"] == bus), False)

    def _db(self, bus):
        return next((s["db"] for s in self.obj.sends if s["bus"] == bus), -12.0)


class MixerView(QWidget):
    def __init__(self, project, session, main):
        super().__init__()
        self.p, self.s, self.main = project, session, main
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        for txt, fn, tt in (("Add Bus", self._bus, "Add a submix bus. Route tracks to it, or send to it for shared reverb and delay."),
                            ("Add VCA", self._vca, "Add a VCA master that scales several faders together.")):
            b = QPushButton(txt)
            b.clicked.connect(lambda _=False, fn=fn: fn())
            tip(b, tt)
            top.addWidget(b)
        top.addWidget(QLabel("Tracks > (sends) > buses > master. Double-click an effect to edit it."))
        top.addStretch(1)
        v.addLayout(top)
        self.sa = QScrollArea()
        self.sa.setWidgetResizable(True)
        self.body = QWidget()
        self.row = QHBoxLayout(self.body)
        self.row.setAlignment(Qt.AlignLeft)
        self.sa.setWidget(self.body)
        v.addWidget(self.sa, 1)
        self.strips, self._sig = [], None
        project.add_listener(self.refresh)
        session.tick.connect(self._tick)
        self.refresh(force=True)

    def _bus(self):
        self.main.attach_project()
        self.p.add_bus()

    def _vca(self):
        self.main.attach_project()
        self.p.add_vca()

    def strip_for(self, kind, obj):
        return next((s for s in self.strips if s.kind == kind and s.obj is obj), self.strips[0] if self.strips else None)

    def signature(self):
        return (tuple((t.id, t.name, t.is_bus, t.output, t.vca, len(t.inserts), tuple(s["bus"] for s in t.sends), tuple(i["bypass"] for i in t.inserts))
                      for t in self.p.tracks), tuple((x["id"], x["name"]) for x in self.p.vcas),
                tuple((i["kind"], i["bypass"]) for i in self.p.master["inserts"]))

    def refresh(self, force=False):
        sig = self.signature()
        if force or sig != self._sig:
            self._sig = sig
            while self.row.count():
                it = self.row.takeAt(0)
                if it.widget():
                    _gone(it.widget())
            self.strips = []
            for t in self.p.tracks:
                st = Strip(self, "bus" if t.is_bus else "track", t)
                self.strips.append(st)
                self.row.addWidget(st)
            for vca in self.p.vcas:
                st = Strip(self, "vca", vca)
                self.strips.append(st)
                self.row.addWidget(st)
            st = Strip(self, "master", self.p.master)
            self.strips.append(st)
            self.row.addWidget(st)
        else:
            for st in self.strips:
                st.sync()

    def _tick(self):
        if self.strips and self.isVisible():
            m = self.strips[-1]
            m.level.setValue(int(max(0, 600 + 10 * 20 * np.log10(max(self.s.player.peak, 1e-4)) * 1.0)))


# ======================================================================== meters
class MetersPanel(QWidget):
    """Momentary / short-term / integrated loudness, loudness range, true peak and a history graph."""

    def __init__(self, session):
        super().__init__()
        self.s = session
        self.data = {}
        v = QVBoxLayout(self)
        self.lab = QLabel("Render a preview to see loudness meters.")
        self.lab.setObjectName("info")
        self.lab.setWordWrap(True)
        self.lab.setStyleSheet("font-size: 14px;")
        v.addWidget(self.lab)
        self.graph = _LoudnessGraph(self)
        v.addWidget(self.graph, 1)
        session.previewReady.connect(self.compute)
        session.audioChanged.connect(self.compute)
        session.tick.connect(self._tick)
        session.stateChanged.connect(self.graph.update)
        self._task = None

    def compute(self):
        s = self.s
        a, b = s.player.a, s.player.b
        if a is None:
            return
        t = Task(lambda prog, cancel: {"A": dsp.meter_analysis(a), "B": dsp.meter_analysis(b) if b is not None else None})
        s._hold(t)
        t.sig.done.connect(self._done)
        t.sig.failed.connect(lambda m: None)
        self._task = t
        s.pool.start(t)

    def _done(self, res):
        self.data = res
        self.graph.update()
        self._tick()

    def cur(self):
        d = self.data.get(self.s.player.src) or self.data.get("A")
        return d

    def _tick(self):
        d = self.cur()
        if not d:
            return
        i = int(min(max(0, (self.s.player.pos / SR) / d["hop"]), len(d["momentary"]) - 1))
        f = lambda x: "-inf" if not np.isfinite(x) else f"{x:.1f}"
        self.lab.setText(f"Momentary {f(d['momentary'][i])} LUFS    Short-term {f(d['short'][min(i, len(d['short']) - 1)])} LUFS    "
                         f"Integrated {f(d['integrated'])} LUFS\nLoudness range {d['lra']:.1f} LU    True peak {d['true_peak']:.1f} dBTP    "
                         f"Sample peak {d['peak']:.1f} dBFS    ({'cleaned (B)' if self.s.player.src == 'B' else 'original (A)'})")
        self.graph.update()


class _LoudnessGraph(QWidget):
    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.setMinimumHeight(90)

    def paintEvent(self, _):
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, theme.color("panel2"))
        d = self.panel.cur()
        if not d:
            return
        lo, hi = -50.0, -5.0
        y = lambda v: H - (min(hi, max(lo, v)) - lo) / (hi - lo) * H
        p.setPen(QPen(theme.color("grid"), 1))
        for db in (-40, -30, -23, -16, -10):
            p.drawLine(0, int(y(db)), W, int(y(db)))
            p.setPen(theme.color("muted"))
            p.drawText(4, int(y(db)) - 2, f"{db}")
            p.setPen(QPen(theme.color("grid"), 1))
        n = len(d["short"])
        for series, col, wd in ((d["momentary"], theme.color("accent", 120), 1.2), (d["short"], theme.color("accent"), 2.2)):
            p.setPen(QPen(col, wd))
            pts = [(k / max(1, len(series) - 1) * W, y(float(v))) for k, v in enumerate(series[:: max(1, len(series) // max(W, 1))])]
            step = max(1, len(series) // max(W, 1))
            xs = [k * step / max(1, len(series) - 1) * W for k in range(len(pts))]
            for k in range(1, len(pts)):
                p.drawLine(int(xs[k - 1]), int(pts[k - 1][1]), int(xs[k]), int(pts[k][1]))
        total = max(1, n * d["hop"])
        x = (self.panel.s.player.pos / SR) / total * W
        p.setPen(QPen(theme.color("text"), 2))
        p.drawLine(int(x), 0, int(x), H)
