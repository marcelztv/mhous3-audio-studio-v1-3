"""The workspaces.  All of them are views onto the same Session, so switching never loses work."""
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, QUrl, Qt, Signal
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDockWidget, QFileDialog, QFormLayout,
                               QFrame, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QTextBrowser, QToolBar, QVBoxLayout, QWidget)

from . import APP_NAME, APP_VERSION, SR, engine, settings as cfg, theme
from .mixer import MetersPanel, MixerView
from .timeline import TimelineEditor, TrackInspector
from .waveform import WaveformPanel
from .widgets import (Knob, LiveSpectrogram, OutputPanel, PresetBar, PresetCards, advanced_panel, basic_panel, hint,
                      scrolled, tip)

MEDIA_FILTER = ("Audio / video (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.wma *.aif *.aiff *.mp4 *.mov *.mkv "
                "*.avi *.mts *.m2ts *.mxf *.webm *.mpg *.mpeg *.wmv *.m4v);;All files (*.*)")
MEDIA_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aif", ".aiff"} | cfg.VIDEO_EXT


def fmt_time(s):
    return f"{int(s // 60)}:{s % 60:04.1f}"


# ======================================================================== shared building blocks
def bind_waveform(panel, session):
    v = panel.view
    v.seekRequested.connect(session.seek_global)
    v.selectionChanged.connect(lambda: session.set_selection(v.sel))
    session.audioChanged.connect(lambda: v.set_orig(session.audio, keep_view=v.orig is not None) if session.audio is not None else None)
    session.loaded.connect(lambda _p: v.set_orig(session.audio))
    session.previewReady.connect(lambda: v.set_clean(session.player.b, session.p_off))
    session.selectionChanged.connect(lambda: v.set_selection(session.sel))

    def tick():
        g = session.global_pos()
        if session.player.playing:
            v.ensure_visible(g)
        v.set_playhead(g)
    session.tick.connect(tick)


class TransportBar(QWidget):
    def __init__(self, session):
        super().__init__()
        self.s = session
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.play = tip(QPushButton("Play"), "Play or pause (Space).")
        self.play.clicked.connect(session.toggle_play)
        stop = tip(QPushButton("Stop"), "Stop and return to the start.")
        stop.clicked.connect(session.stop)
        self.a = tip(QPushButton("A  Original"), "Hear the original recording (key: A).")
        self.b = tip(QPushButton("B  Cleaned"), "Hear the cleaned version (key: B). Switching is instant, so you can compare.")
        for b in (self.a, self.b):
            b.setCheckable(True)
            b.setMinimumWidth(112)
        self.a.clicked.connect(lambda: session.set_ab("A"))
        self.b.clicked.connect(lambda: session.set_ab("B"))
        self.match = tip(QCheckBox("Match loudness"), "Plays the original at the same loudness as the cleaned version so you compare quality, not volume.")
        self.match.setChecked(True)
        self.match.toggled.connect(session.set_match)
        self.loop = tip(QCheckBox("Loop selection"), "Repeat the region you selected on the waveform.")
        self.loop.toggled.connect(session.set_loop)
        self.time = QLabel("0:00.0 / 0:00.0")
        self.meter = tip(QProgressBar(), "Output level meter.")
        self.meter.setRange(0, 60)
        self.meter.setTextVisible(False)
        self.meter.setFixedWidth(90)
        for w in (self.play, stop, self.a, self.b, self.match, self.loop):
            lay.addWidget(w)
        lay.addStretch(1)
        lay.addWidget(self.time)
        lay.addWidget(self.meter)
        session.tick.connect(self._tick)
        session.stateChanged.connect(self._state)

    def _state(self):
        src = self.s.player.src
        self.a.setChecked(src == "A")
        self.b.setChecked(src == "B")

    def _tick(self):
        if self.s.audio is None:
            return
        self.play.setText("Pause" if self.s.player.playing else "Play")
        self.time.setText(f"{fmt_time(self.s.global_pos() / SR)} / {fmt_time(self.s.n / SR)}")
        pk = 20 * np.log10(max(self.s.player.peak, 1e-4))
        self.meter.setValue(int(max(0, 60 + pk)))


class StatusStrip(QWidget):
    def __init__(self, session):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFormat("Ready")
        self.info = QLabel("Open a file to begin. You can also drag one onto the window.")
        self.info.setObjectName("info")
        self.info.setWordWrap(True)
        lay.addWidget(self.bar)
        lay.addWidget(self.info)
        session.progress.connect(lambda f, m: (self.bar.setValue(int(f * 1000)), self.bar.setFormat(f"{m}...  %p%")))
        session.previewStarted.connect(lambda: self.bar.setFormat("Rendering preview..."))
        session.previewReady.connect(lambda: (self.bar.setValue(1000), self.bar.setFormat("Preview ready")))
        session.previewFailed.connect(lambda m: (self.bar.setValue(0), self.bar.setFormat("Ready" if m == "Cancelled" else "Preview failed")))
        session.info.connect(self.info.setText)


def edit_toolbar(session, main, vertical=True):
    tb = QToolBar("Tools")
    tb.setObjectName("tools")
    tb.setOrientation(Qt.Vertical if vertical else Qt.Horizontal)
    tb.setMovable(True)
    items = [("Trim", session.trim, "Trim: keep only the selected region (select on the waveform first)."),
             ("Cut", session.cut, "Cut: delete the selected region."),
             ("Silence", session.silence, "Silence: mute the selected region."),
             ("Fade In", lambda: session.fade(True), "Fade in across the selection, or the whole file."),
             ("Fade Out", lambda: session.fade(False), "Fade out across the selection, or the whole file."),
             ("Gain...", main.ask_gain, "Raise or lower volume of the selection, or the whole file."),
             ("Normalize", session.normalize, "Set the loudest peak to -1 dB."),
             ("Reverse", session.reverse, "Play the selection backwards."),
             ("Undo", session.do_undo, "Undo the last edit."), ("Redo", session.do_redo, "Redo the edit you undid.")]
    for txt, fn, tt in items:
        a = QAction(txt, tb)
        a.setToolTip(tt)
        a.setStatusTip(tt)
        a.triggered.connect(lambda _=False, fn=fn: fn())
        tb.addAction(a)
    return tb


def make_dock(win, title, widget, area=Qt.RightDockWidgetArea, name=None):
    d = QDockWidget(title, win)
    d.setObjectName(name or title.replace(" ", "_"))
    d.setWidget(widget)
    d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
    if area == Qt.RightDockWidgetArea:
        d.setMinimumWidth(340)
    win.addDockWidget(area, d)
    return d


class WorkspacePage(QMainWindow):
    """Base: a QMainWindow embedded as a page, with a saved/restorable dock layout."""
    NAME = ""

    def __init__(self, session, store, main):
        super().__init__()
        self.session, self.store, self.main = session, store, main
        self.setDockNestingEnabled(True)
        self.docks = {}
        self._default_state = None

    def finish(self):
        self._default_state = self.saveState()

    def on_show(self):
        pass

    def save_layout(self):
        return bytes(self.saveState().toBase64()).decode()

    def restore_layout(self, b64):
        from PySide6.QtCore import QByteArray
        if b64:
            self.restoreState(QByteArray.fromBase64(b64.encode()))

    def reset_layout(self):
        if self._default_state is not None:
            self.restoreState(self._default_state)
            for d in self.docks.values():
                d.show()

    def tabify_all(self, names):
        first = self.docks[names[0]]
        for n in names[1:]:
            self.tabifyDockWidget(first, self.docks[n])
        first.raise_()


# ======================================================================== FastTrack
class FastTrackPage(WorkspacePage):
    NAME = "FastTrack"

    def __init__(self, session, store, main):
        super().__init__(session, store, main)
        c = QWidget()
        v = QVBoxLayout(c)
        top = QHBoxLayout()
        self.open_btn = tip(QPushButton("Open file"), "Choose an audio or video file to clean up.")
        self.open_btn.clicked.connect(main.open_dialog)
        self.title = QLabel("No file loaded")
        self.title.setObjectName("hint")
        self.auto_btn = tip(QPushButton("Auto Clean"), "One click: we listen to your file, then you choose a preset or let us pick the best settings.")
        self.auto_btn.setObjectName("hero")
        self.auto_btn.clicked.connect(main.auto_clean)
        top.addWidget(self.open_btn)
        top.addWidget(self.title, 1)
        top.addWidget(self.auto_btn)
        v.addLayout(top)

        self.spec = tip(LiveSpectrogram(session), "Live spectrogram: shows the sound you are hearing, scrolling as it plays. Low sounds at the bottom, high at the top.")
        self.wave = WaveformPanel()
        bind_waveform(self.wave, session)
        self.wave.view.setMinimumHeight(110)
        sp = QSplitter(Qt.Vertical)
        sp.addWidget(self.spec)
        sp.addWidget(self.wave)
        sp.setSizes([260, 150])
        v.addWidget(sp, 1)
        self.transport = TransportBar(session)
        v.addWidget(self.transport)

        knobs = QGroupBox("Quick controls")
        kl = QHBoxLayout(knobs)
        self.knobs = [
            Knob(store, "volume_boost", "Volume Boost", -12, 24, 0.5, " dB", 1, 0.0, "Volume Boost: make the whole recording louder (or quieter). Peaks are protected, so it will not distort."),
            Knob(store, "nr_amount", "Noise Reduction", 0, 100, 1, " %", 0, None, "Noise Reduction: remove hiss, hum, fans and background noise."),
            Knob(store, "derev_amount", "Echo Removal", 0, 100, 1, " %", 0, None, "Echo Removal: reduce the hollow sound of a room."),
            Knob(store, "room_expansion", "Room Expansion", 0, 100, 1, " %", 0, None, "Room Expansion: give a dry voice a natural sense of space and width."),
            Knob(store, "reverb_mix", "Reverb", 0, 100, 1, " %", 0, None, "Reverb: add a room's echo on purpose. Choose the room type in More Options."),
        ]
        for k in self.knobs:
            kl.addWidget(k)
        v.addWidget(knobs)
        v.addWidget(QLabel("Presets"))
        self.cards = PresetCards(main.user_presets, horizontal=True)
        self.cards.picked.connect(main.pick_preset)
        v.addWidget(self.cards)
        v.addWidget(StatusStrip(session))
        self.setCentralWidget(c)

        self.docks["Output"] = make_dock(self, "Output", scrolled(self._output()))
        self.docks["More Options"] = make_dock(self, "More Options", scrolled(basic_panel(store, main.pick_preset, main.user_presets)))
        self.tabify_all(["More Options", "Output"])
        self.docks["More Options"].raise_()
        session.loaded.connect(lambda p: self.title.setText(f"{Path(p).name}   -   {session.n / SR:.1f} s"))
        session.audioChanged.connect(lambda: self.title.setText(f"{Path(session.src_path).name} (edited)   -   {session.n / SR:.1f} s") if session.src_path and not session.undo == [] else None)
        session.stateChanged.connect(lambda: self.cards.select(session.cleaned_by or ""))
        self.finish()

    def _output(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(OutputPanel(self.store))
        l.addStretch(1)
        return w


# ======================================================================== Precision
class PrecisionPage(WorkspacePage):
    NAME = "Precision"

    def __init__(self, session, store, main, full=False):
        super().__init__(session, store, main)
        self.full = full
        c = QWidget()
        v = QVBoxLayout(c)
        if full:
            banner = QLabel("Full Studio: multi-track timeline, mixer, 33 effects, JSFX scripting, loudness meters and session export (RPP, AAF, EDL, "
                            "Final Cut XML) are all here. Still to come: live automation recording, MIDI and spatial tools.")
            banner.setObjectName("info")
            banner.setWordWrap(True)
            v.addWidget(banner)
        self.tabs = QTabWidget()
        # ---- multi-track
        self.timeline = TimelineEditor(main.project, session, store, main)
        self.live = tip(LiveSpectrogram(session), "Live spectrogram: scrolls as the audio plays. Shows the version you are listening to (A or B).")
        mt = QSplitter(Qt.Vertical)
        mt.addWidget(self.timeline)
        mt.addWidget(self.live)
        mt.setSizes([430, 170])
        self.tabs.addTab(mt, "Multi-track")
        self.mixer = MixerView(main.project, session, main)
        self.tabs.addTab(self.mixer, "Mixer")
        # ---- single-file waveform editor (everything from the earlier versions)
        w = QWidget()
        wl = QVBoxLayout(w)
        top = QHBoxLayout()
        self.title = QLabel("No file loaded")
        self.title.setObjectName("hint")
        top.addWidget(self.title, 1)
        self.spec_btn = tip(QPushButton("Full-file spectrogram"), "Switch the waveform area between waveform and a detailed spectrogram of the whole file.")
        self.spec_btn.setCheckable(True)
        top.addWidget(self.spec_btn)
        self.wave = WaveformPanel()
        bind_waveform(self.wave, session)
        self.spec_btn.toggled.connect(lambda on: self.wave.view.set_mode("spec" if on else "wave"))
        for txt, fn, tt in (("Fit", lambda: self.wave.view.zoom_fit(), "Show the whole file."),
                            ("+", lambda: self.wave.view.zoom(0.6), "Zoom in."), ("-", lambda: self.wave.view.zoom(1.6), "Zoom out."),
                            ("Zoom selection", lambda: self.wave.view.zoom_to_selection(), "Zoom to the selected region.")):
            b = tip(QPushButton(txt), tt)
            b.clicked.connect(fn)
            top.addWidget(b)
        wl.addLayout(top)
        self.live2 = tip(LiveSpectrogram(session), "Live spectrogram: scrolls as the audio plays.")
        sp = QSplitter(Qt.Vertical)
        sp.addWidget(self.wave)
        sp.addWidget(self.live2)
        sp.setSizes([360, 200])
        wl.addWidget(sp, 1)
        self.tabs.addTab(w, "Waveform editor")
        v.addWidget(self.tabs, 1)
        self.transport = TransportBar(session)
        v.addWidget(self.transport)
        v.addWidget(StatusStrip(session))
        self.setCentralWidget(c)

        self.tools = edit_toolbar(session, main)
        self.addToolBar(Qt.LeftToolBarArea, self.tools)
        self.tools.setVisible(False)
        self.docks["Essentials"] = make_dock(self, "Essentials", scrolled(basic_panel(store, main.pick_preset, main.user_presets)))
        self.docks["Pro Controls"] = make_dock(self, "Pro Controls", scrolled(advanced_panel(store)))
        self.docks["Inspector"] = make_dock(self, "Inspector", scrolled(TrackInspector(main.project, self.timeline)))
        self.preset_cards = PresetCards(main.user_presets, horizontal=False)
        self.preset_cards.picked.connect(main.pick_preset)
        self.docks["Presets"] = make_dock(self, "Presets", self.preset_cards)
        names = ["Inspector", "Essentials", "Pro Controls", "Presets"]
        self.docks["Output"] = make_dock(self, "Output", scrolled(self._output()))
        names.append("Output")
        self.meters = MetersPanel(session)
        if full:
            self.docks["Meters"] = make_dock(self, "Meters", self.meters, Qt.BottomDockWidgetArea)
        else:
            self.docks["Meters"] = make_dock(self, "Meters", self.meters)
            names.append("Meters")
        self.tabify_all(names)
        self.docks["Inspector"].raise_()
        session.loaded.connect(lambda p: self.title.setText(f"{Path(p).name}   -   {session.n / SR:.1f} s"))
        session.stateChanged.connect(lambda: self.preset_cards.select(session.cleaned_by or ""))
        self.tabs.currentChanged.connect(self._tab)
        self.finish()

    def _tab(self, i):
        self.tools.setVisible(i == 2)
        if i in (0, 1):
            self.timeline.ensure_loaded()
        if i == 1:
            self.mixer.refresh(force=True)

    def on_show(self):
        if self.tabs.currentIndex() == 0:
            self.timeline.ensure_loaded()

    def _output(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(OutputPanel(self.store))
        l.addStretch(1)
        return w

    def _meters(self):
        w = QWidget()
        l = QHBoxLayout(w)
        self.meter_lbl = QLabel("Process a file to see loudness and peak readings.")
        self.meter_lbl.setObjectName("info")
        self.meter_lbl.setWordWrap(True)
        l.addWidget(self.meter_lbl)

        def upd():
            info = self.session.preview[1] if self.session.preview else None
            if info:
                self.meter_lbl.setText(f"Integrated loudness: in {info['lufs_in']:.1f} LUFS, out {info['lufs_out']:.1f} LUFS   |   "
                                       f"Peak out {info['peak_out_db']:.1f} dBFS   |   Channels {info['channels']}")
        self.session.previewReady.connect(upd)
        return w


class FullStudioPage(PrecisionPage):
    NAME = "Full Studio"

    def __init__(self, session, store, main):
        super().__init__(session, store, main, full=True)


# ======================================================================== Auto Clean dialog
class AutoCleanDialog(QDialog):
    def __init__(self, parent, scenario, notes, user_presets):
        super().__init__(parent)
        self.setWindowTitle("Auto Clean")
        self.setMinimumSize(720, 560)
        self.choice = "auto"
        v = QVBoxLayout(self)
        h = QLabel("How should we clean this audio?")
        h.setStyleSheet("font-size: 20px; font-weight: 700;")
        v.addWidget(h)
        v.addWidget(hint("We listened to your file. Use our recommendation, or pick a preset."))
        self.rec = QFrame()
        self.rec.setObjectName("card")
        rl = QVBoxLayout(self.rec)
        t = QLabel(f"Recommended for this audio: {scenario}")
        t.setObjectName("cardtitle")
        t.setStyleSheet("font-size: 15px;")
        rl.addWidget(t)
        for n in notes:
            lab = QLabel("- " + n)
            lab.setWordWrap(True)
            lab.setObjectName("carddesc")
            rl.addWidget(lab)
        self.rec.setProperty("selected", True)
        self.rec.setCursor(Qt.PointingHandCursor)
        tip(self.rec, "Use the settings we chose after listening to your file.")
        self.rec.mousePressEvent = lambda e: self._pick("auto")
        v.addWidget(self.rec)
        v.addWidget(QLabel("Or choose a preset instead:"))
        self.cards = PresetCards(user_presets, horizontal=False, include_auto=False)
        self.cards.picked.connect(self._pick)
        v.addWidget(self.cards, 1)
        self.always = tip(QCheckBox("Always use the recommended settings without asking"), "Skip this window next time and clean automatically.")
        v.addWidget(self.always)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Clean now")
        ok.setObjectName("primary")
        ok.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(ok)
        v.addLayout(row)

    def _pick(self, name):
        self.choice = name
        self.rec.setProperty("selected", name == "auto")
        self.rec.style().unpolish(self.rec)
        self.rec.style().polish(self.rec)
        self.cards.select(name if name != "auto" else "")


# ======================================================================== Batch
class BatchWorker(QThread):
    row = Signal(int, str, str, str)
    prog = Signal(float, str)
    finished_all = Signal(str)

    def __init__(self, jobs, s, outdir, suffix):
        super().__init__()
        self.jobs, self.s, self.outdir, self.suffix = jobs, s, outdir, suffix
        self.cancel = threading.Event()

    def run(self):
        n, ok, bad = len(self.jobs), 0, 0
        for i, (r, src) in enumerate(self.jobs):
            if self.cancel.is_set():
                break
            self.row.emit(r, "Processing...", "", "")
            try:
                dst = engine.default_output_path(src, self.s, self.outdir or None, self.suffix)
                info = engine.process_file(src, dst, self.s, lambda f, m, i=i, src=src: self.prog.emit(
                    (i + f) / n, f"{Path(src).name}: {m}"), self.cancel)
                self.row.emit(r, "Done", f"{info['lufs_in']:.1f} -> {info['lufs_out']:.1f} LUFS", str(dst))
                ok += 1
            except engine.Cancelled:
                self.row.emit(r, "Cancelled", "", "")
                break
            except Exception as e:
                self.row.emit(r, "Failed", str(e)[:200], "")
                bad += 1
        self.finished_all.emit(f"Finished: {ok} done, {bad} failed" + (", stopped early" if self.cancel.is_set() else ""))


class BatchPage(QWidget):
    def __init__(self, store, main):
        super().__init__()
        self.store, self.main, self.worker, self.out_dir = store, main, None, ""
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        root.addLayout(left, 3)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["File", "Status", "Result", "Saved as"])
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tip(self.table, "Your queue. Drag files or folders here to add them.")
        left.addWidget(self.table, 1)
        bar = QHBoxLayout()
        for txt, fn, tt in (("Add files...", self.add_files, "Add audio or video files to the queue."),
                            ("Add folder...", self.add_folder, "Add every audio/video file in a folder."),
                            ("Remove selected", self.remove, "Remove highlighted rows."),
                            ("Clear", lambda: self.table.setRowCount(0), "Empty the queue.")):
            b = tip(QPushButton(txt), tt)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch(1)
        left.addLayout(bar)
        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        self.prog.setFormat("Ready")
        left.addWidget(self.prog)
        self.summary = QLabel("Drag and drop files or folders anywhere on this page.")
        self.summary.setObjectName("info")
        self.summary.setWordWrap(True)
        left.addWidget(self.summary)

        right = QVBoxLayout()
        root.addLayout(right, 2)
        box = QGroupBox("Batch settings")
        f = QFormLayout(box)
        self.which = tip(QComboBox(), "Which settings to use: whatever is set now, or a preset.")
        f.addRow("Settings to use", self.which)
        self.keep_out = tip(QCheckBox("Use my Output choices below"), "Keep your channels, format and loudness even when a preset is chosen.")
        self.keep_out.setChecked(True)
        f.addRow(self.keep_out)
        self.same = tip(QCheckBox("Save next to the original file"), "Put each result beside its source file.")
        self.same.setChecked(True)
        self.same.toggled.connect(lambda v: self.folder_btn.setEnabled(not v))
        f.addRow(self.same)
        self.folder = QLabel("(not chosen)")
        self.folder.setObjectName("hint")
        self.folder_btn = tip(QPushButton("Choose output folder..."), "Save all results in one folder.")
        self.folder_btn.setEnabled(False)
        self.folder_btn.clicked.connect(self._choose_folder)
        f.addRow(self.folder_btn)
        f.addRow(self.folder)
        self.suffix = tip(QComboBox(), "Text added to each result's file name.")
        self.suffix.setEditable(True)
        self.suffix.addItems(["_MHous3", "_clean", "_cinema", ""])
        f.addRow("File name ending", self.suffix)
        right.addWidget(box)
        right.addWidget(OutputPanel(store))
        self.btn_start = tip(QPushButton("Start batch"), "Process every file in the queue, one after another.")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = tip(QPushButton("Stop"), "Stop after the current stage.")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self.worker and self.worker.cancel.set())
        btn_open = tip(QPushButton("Open output folder"), "Open the folder with your results.")
        btn_open.clicked.connect(self._open_folder)
        for b in (self.btn_start, self.btn_stop, btn_open):
            right.addWidget(b)
        right.addStretch(1)
        self.setAcceptDrops(True)
        self.refresh_choices()
        store.changed.connect(lambda k: self.refresh_choices() if k == "__presets__" else None)

    def refresh_choices(self):
        cur = self.which.currentText()
        self.which.clear()
        self.which.addItem("Current settings")
        self.which.addItems(list(cfg.PRESETS) + ["* " + n for n in self.main.user_presets])
        self.which.setCurrentIndex(max(self.which.findText(cur), 0))

    def _choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder")
        if d:
            self.out_dir = d
            self.folder.setText(d)

    def add_files(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Add files", "", MEDIA_FILTER)
        self.add_paths(fs)

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder")
        if d:
            self.add_paths([d])

    def add_paths(self, paths):
        have = {self.table.item(r, 0).data(Qt.UserRole) for r in range(self.table.rowCount())}
        for p in paths:
            pp = Path(p)
            if pp.is_dir():
                self.add_paths([str(x) for x in sorted(pp.rglob("*")) if x.suffix.lower() in MEDIA_EXT])
                continue
            if p in have or pp.suffix.lower() not in MEDIA_EXT:
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            it = QTableWidgetItem(pp.name)
            it.setData(Qt.UserRole, str(p))
            it.setToolTip(str(p))
            self.table.setItem(r, 0, it)
            for c, t in ((1, "Waiting"), (2, ""), (3, "")):
                self.table.setItem(r, c, QTableWidgetItem(t))
        self.summary.setText(f"{self.table.rowCount()} file(s) in the queue.")

    def remove(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    def _settings(self):
        name = self.which.currentText()
        if name == "Current settings":
            return dict(self.store.s)
        s = self.main.preset_dict(name, keep_output=False)
        if self.keep_out.isChecked():
            for k in ("out_channels", "out_format", "loudness_on", "loudness_target", "lufs_custom", "ceiling"):
                s[k] = self.store.s[k]
        return s

    def start(self):
        jobs = [(r, self.table.item(r, 0).data(Qt.UserRole)) for r in range(self.table.rowCount())
                if self.table.item(r, 1).text() != "Done"]
        if not jobs:
            QMessageBox.information(self, APP_NAME, "Add some files first (or everything is already done).")
            return
        if not self.same.isChecked() and not self.out_dir:
            self._choose_folder()
            if not self.out_dir:
                return
        self.worker = BatchWorker(jobs, self._settings(), "" if self.same.isChecked() else self.out_dir, self.suffix.currentText())
        self.worker.row.connect(self._row)
        self.worker.prog.connect(lambda f, m: (self.prog.setValue(int(f * 1000)), self.prog.setFormat(f"{m}  %p%")))
        self.worker.finished_all.connect(self._finished)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.worker.start()
        self.worker.setPriority(QThread.LowPriority)

    def _row(self, r, status, result, saved):
        for c, t in ((1, status), (2, result), (3, saved)):
            if r < self.table.rowCount():
                self.table.item(r, c).setText(t)

    def _finished(self, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.summary.setText(msg)
        self.prog.setFormat(msg)

    def _open_folder(self):
        for r in range(self.table.rowCount()):
            p = self.table.item(r, 3).text()
            if p:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(p).parent)))
                return
        if self.out_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir))


# ======================================================================== Guide
def guide_html():
    p = theme.palette()
    a = p["accent"]
    return f"""
<h2 style='color:{a}'>Welcome to {APP_NAME}</h2>
<p>Built for M Hous3 Media. Free, offline audio rescue and cinema-style finishing. Nothing is uploaded anywhere.</p>
<h3 style='color:{a}'>Three workspaces</h3>
<ul>
<li><b>FastTrack</b> - for fast, great results with no sound-design knowledge: Auto Clean, big knobs, preset cards.</li>
<li><b>Precision</b> - focused manual control: a multi-track timeline (clips, fades, crossfades, trims, clip gain, volume automation, stems), a single-file waveform editor, full parameter sliders and docked panels you can rearrange.</li>
<li><b>Full Studio</b> - the professional environment: timeline, mixer, effects, JSFX, meters and session export. Live automation recording, MIDI and spatial tools are still to come.</li>
</ul>
<h3 style='color:{a}'>Quick start</h3>
<ol>
<li>Click <b>Open file</b> (or drag a file in).</li>
<li>Click <b>Auto Clean</b>. Pick a preset or accept the recommendation.</li>
<li>Press <b>Space</b> to play. Press <b>A</b> and <b>B</b> to compare original and cleaned.</li>
<li>Turn <b>Volume Boost</b>, <b>Noise Reduction</b>, <b>Echo Removal</b>, <b>Room Expansion</b> or <b>Reverb</b> until it sounds right.</li>
<li><b>Export</b>, or use <b>Batch</b> for many files.</li>
</ol>
<h3 style='color:{a}'>Multi-track timeline</h3>
<p>Open a file, then switch to <b>Precision &gt; Multi-track</b>: it appears as Track 1. Drop more files on the timeline to add tracks. Drag clips to move them, drag clip edges to trim, drag the top corners for fades, drag the dashed line for clip gain, and click in the automation lane (button <b>A</b>) to draw volume curves. Use <b>Separate</b>, <b>Duplicate</b>, <b>Repeat</b>, <b>Fades</b> and <b>Clip</b> menus for the rest. <b>File &gt; Save project</b> copies all media into a project folder, and <b>Stems</b> exports every track.</p>
<h3 style='color:{a}'>Mixer, effects and scripting</h3>
<p>The <b>Mixer</b> tab shows a strip for every track, bus and VCA plus the master: faders, pan, <b>sends</b> (pre or post fader), <b>insert effects</b> and output routing. Add any of 33 effects (dynamics, reverbs, delays, modulation, EQ and filters, pitch, distortion, utility) or write your own <b>JSFX</b> script; double-click an effect to edit it. <b>Track presets</b> load ready-made chains, and <b>File &gt; New project from template</b> builds a session for podcasts, dialogue, music and more. The <b>Meters</b> panel shows momentary, short-term and integrated loudness, loudness range and true peak. <b>Macros</b> let you chain editing commands, and <b>Customize &gt; Session settings</b> switches the ruler between time, timecode and bars | beats. Sessions can be exported as REAPER (.rpp), AAF, EDL and Final Cut XML, and imported from RPP and Final Cut XML.</p>
<h3 style='color:{a}'>Refining an automatic clean</h3>
<p>After Auto Clean or a preset, you can keep refining in Precision or Full Studio. When you switch, you are asked whether to <b>keep your changes</b> or <b>start again from the original</b>.</p>
<h3 style='color:{a}'>Make it yours</h3>
<p>Drag, float and hide panels. Use <b>Customize &gt; Keyboard Shortcuts</b> to change keys for each workspace. <b>View &gt; Theme</b> switches Dark, Light or System. The tutorial can be replayed any time from the <b>Tutorial</b> button.</p>
<p style='color:{p["muted"]}'>Version {APP_VERSION}. (c) M Hous3 Media. Uses ffmpeg and DeepFilterNet, which ship inside this app.</p>
"""


class GuidePage(QTextBrowser):
    def __init__(self):
        super().__init__()
        self.refresh()

    def refresh(self):
        self.setHtml(guide_html())
