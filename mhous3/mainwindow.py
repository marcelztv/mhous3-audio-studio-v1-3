"""MHous3 Audio Studio main window: Photoshop-style ribbon, three workspaces, Batch and Guide."""
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (QFormLayout, QMenu, QDoubleSpinBox, QSpinBox, QApplication, QComboBox, QDialog, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QMainWindow, QMessageBox, QPushButton, QStackedWidget, QTableWidget, QTableWidgetItem,
                               QToolBar, QToolButton, QVBoxLayout, QWidget, QKeySequenceEdit, QHeaderView, QSizePolicy)

from . import APP_NAME, APP_VERSION, analyzer, engine, settings as cfg, theme
from .pages import (AutoCleanDialog, BatchPage, FastTrackPage, FullStudioPage, GuidePage, MEDIA_FILTER, PrecisionPage)
from . import fx, interchange as ix
from .macros import MacroDialog, run_macro
from .paths import app_dir
from .project import Project
from .session import Session, Task
from .store import Store
from .tutorial import TutorialOverlay
from .widgets import PresetCards, tip

OUT_KEYS = ("out_channels", "out_format", "loudness_on", "loudness_target", "lufs_custom", "ceiling")

WS_TIPS = {
    "FastTrack": "Workspace: FastTrack - Designed for fast, high-quality asset delivery. Streamlines processing into automatic "
                 "tools, core presets, and simple leveling knobs.",
    "Precision": "Workspace: Precision - Provides focused manual control. Opens advanced parameter sliders, detailed editing "
                 "tools and a high-resolution spectrogram. (Multi-track lanes arrive in the next update.)",
    "Full Studio": "Workspace: Full Studio - The professional production environment: multi-track timeline, console-style mixer, 33 effects "
                   "plus JSFX scripting, loudness meters and session export (RPP, AAF, EDL, Final Cut XML). Still in development: live "
                   "automation recording, MIDI and spatial tools.",
}

ACTIONS = [  # id, label, default shortcut
    ("open", "Open file", "Ctrl+O"), ("export", "Export", "Ctrl+E"), ("auto", "Auto Clean", "Ctrl+Shift+A"),
    ("play", "Play / Pause", "Space"), ("stop", "Stop", "Ctrl+."), ("hear_a", "Hear original (A)", "A"),
    ("hear_b", "Hear cleaned (B)", "B"), ("undo", "Undo", "Ctrl+Z"), ("redo", "Redo", "Ctrl+Y"),
    ("ws1", "FastTrack workspace", "Ctrl+1"), ("ws2", "Precision workspace", "Ctrl+2"),
    ("ws3", "Full Studio workspace", "Ctrl+3"), ("batch", "Batch", "Ctrl+4"),
    ("trim", "Trim to selection", "Ctrl+T"), ("cut", "Cut selection", "Ctrl+X"), ("silence", "Silence selection", "Ctrl+Shift+M"),
    ("fade_in", "Fade in", "Ctrl+Alt+I"), ("fade_out", "Fade out", "Ctrl+Alt+O"), ("normalize", "Normalize", "Ctrl+Shift+N"),
    ("zoom_fit", "Zoom to fit", "Ctrl+0"), ("theme", "Cycle theme", "Ctrl+Shift+T"),
    ("tut_basic", "Basic tutorial", "F1"), ("tut_adv", "Advanced tutorial", "Shift+F1"),
    ("clip_copy", "Copy clips", "Ctrl+C"), ("clip_paste", "Paste clips", "Ctrl+V"), ("clip_clear", "Clear clips", "Delete"),
    ("clip_dup", "Duplicate clips", "Ctrl+D"), ("clip_repeat", "Repeat clips", "Ctrl+R"), ("sep", "Separate at playhead", "S"),
    ("nudge_l", "Nudge clips earlier", "Alt+Left"), ("nudge_r", "Nudge clips later", "Alt+Right"),
    ("save_proj", "Save project", "Ctrl+S"), ("open_proj", "Open project", "Ctrl+Shift+O"), ("stems", "Render stems", "Ctrl+Shift+E"),
]


class ShortcutDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(560, 640)
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Workspace:"))
        self.ws = QComboBox()
        self.ws.addItems(["FastTrack", "Precision", "Full Studio"])
        self.ws.setCurrentIndex(min(win.stack.currentIndex(), 2))
        self.ws.currentIndexChanged.connect(self.fill)
        row.addWidget(self.ws, 1)
        v.addLayout(row)
        v.addWidget(QLabel("Click a box and press the new key combination. Each workspace keeps its own set."))
        self.table = QTableWidget(len(ACTIONS), 2)
        self.table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 210)
        self.table.verticalHeader().hide()
        v.addWidget(self.table, 1)
        self.edits = {}
        row = QHBoxLayout()
        reset = tip(QPushButton("Reset this workspace"), "Restore the default keys for the chosen workspace.")
        reset.clicked.connect(self.reset)
        ok = QPushButton("Save")
        ok.setObjectName("primary")
        ok.clicked.connect(self.save)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(reset)
        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(ok)
        v.addLayout(row)
        self.fill()

    def fill(self):
        ws = self.ws.currentText()
        saved = self.win.cfg.get("shortcuts", {}).get(ws, {})
        for r, (aid, label, default) in enumerate(ACTIONS):
            self.table.setItem(r, 0, QTableWidgetItem(label))
            e = QKeySequenceEdit(QKeySequence(saved.get(aid, default)))
            self.table.setCellWidget(r, 1, e)
            self.edits[aid] = e

    def reset(self):
        for aid, label, default in ACTIONS:
            self.edits[aid].setKeySequence(QKeySequence(default))

    def save(self):
        allc = dict(self.win.cfg.get("shortcuts", {}))
        allc[self.ws.currentText()] = {aid: self.edits[aid].keySequence().toString() for aid, _, _ in ACTIONS}
        self.win.cfg.set("shortcuts", allc)
        self.win.apply_shortcuts()
        self.accept()


class SessionDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Session settings")
        self.setMinimumWidth(460)
        st = win.project.settings
        f = QFormLayout()
        v = QVBoxLayout(self)
        self.ruler = QComboBox()
        self.ruler.addItems(["Min:Sec", "Timecode", "Bars|Beats"])
        self.ruler.setCurrentText(st.get("ruler", "Min:Sec"))
        tip(self.ruler, "What the timeline ruler shows: minutes and seconds, video timecode, or bars | beats | ticks (960 per beat).")
        self.fps = QComboBox()
        self.fps.addItems(list(ix.FPS))
        self.fps.setCurrentText(st.get("fps", "25"))
        tip(self.fps, "Frame rate for timecode and for EDL / Final Cut XML export.")
        self.tempo = QDoubleSpinBox()
        self.tempo.setRange(20, 300)
        self.tempo.setValue(float(st.get("tempo", 120)))
        self.sig = QSpinBox()
        self.sig.setRange(1, 16)
        self.sig.setValue(int(st.get("sig_num", 4)))
        self.buf = QComboBox()
        self.buf.addItems(["256", "512", "1024", "2048", "4096"])
        self.buf.setCurrentText(str(win.cfg.get("buffer", 1024)))
        tip(self.buf, "Playback buffer in samples. Smaller = lower delay, larger = fewer glitches on slow computers.")
        f.addRow("Timeline ruler", self.ruler)
        f.addRow("Frame rate", self.fps)
        f.addRow("Tempo (BPM)", self.tempo)
        f.addRow("Beats per bar", self.sig)
        f.addRow("Playback buffer (samples)", self.buf)
        v.addLayout(f)
        v.addWidget(QLabel("Sample rate and bit depth are chosen in the Output panel. Processing always runs at 48 kHz."))
        row = QHBoxLayout()
        row.addStretch(1)
        c = QPushButton("Cancel")
        c.clicked.connect(self.reject)
        ok = QPushButton("OK")
        ok.setObjectName("primary")
        ok.clicked.connect(self._ok)
        row.addWidget(c)
        row.addWidget(ok)
        v.addLayout(row)

    def _ok(self):
        st = self.win.project.settings
        st.update(ruler=self.ruler.currentText(), fps=self.fps.currentText(), tempo=self.tempo.value(), sig_num=self.sig.value())
        b = int(self.buf.currentText())
        self.win.cfg.set("buffer", b)
        pl = self.win.session.player
        if pl.blocksize != b:
            pl.close()
            pl.blocksize = b
        for pg in (self.win.prec, self.win.full):
            pg.timeline.apply_grid()
            pg.timeline.view.update()
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self, config, progress=lambda m: None):
        super().__init__()
        self.cfg = config
        self.setWindowTitle(APP_NAME)
        self.resize(1440, 900)
        self.setAcceptDrops(True)
        ico = app_dir() / "assets" / "icon.png"
        if ico.exists():
            self.setWindowIcon(QIcon(str(ico)))
        self.store = Store()
        self.store.load(cfg.preset_settings("Gentle (keep it natural)"))
        self.user_presets = cfg.load_user_presets()
        self.session = Session(self.store)
        self.session.player.blocksize = int(self.cfg.get("buffer", 1024))
        self.project = Project()
        self.session.loaded.connect(lambda _p: self.project.clear())
        self.acts = {}
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        progress("Building FastTrack workspace...")
        self.fast = FastTrackPage(self.session, self.store, self)
        progress("Building Precision workspace...")
        self.prec = PrecisionPage(self.session, self.store, self)
        progress("Building Full Studio workspace...")
        self.full = FullStudioPage(self.session, self.store, self)
        self.batch = BatchPage(self.store, self)
        self.guide = GuidePage()
        for pg in (self.fast, self.prec, self.full, self.batch, self.guide):
            self.stack.addWidget(pg)
        progress("Assembling the ribbon and menus...")
        self._make_actions()
        self._ribbon()
        self._menus()
        self.apply_shortcuts()
        self.tool_label = QLabel()
        self.statusBar().addPermanentWidget(self.tool_label)
        self._tools_status()
        self.statusBar().showMessage("Ready")
        self.tutorial = TutorialOverlay(self)
        self.tutorial.finished.connect(self._tutorial_done)
        s = self.session
        s.busy.connect(lambda m: (self.statusBar().showMessage(m), QApplication.setOverrideCursor(Qt.BusyCursor)))
        s.idle.connect(lambda: (QApplication.restoreOverrideCursor(), self.statusBar().showMessage("Ready")))
        s.error.connect(lambda m: QMessageBox.warning(self, APP_NAME, m))
        s.exported.connect(lambda dst, info: QMessageBox.information(
            self, APP_NAME, f"Saved:\n{dst}\n\nLoudness {info['lufs_in']:.1f} -> {info['lufs_out']:.1f} LUFS, peak {info['peak_out_db']:.1f} dBFS"))
        for pg in self._pages():
            pg.restore_layout(self.cfg.get(f"layout_{pg.NAME}"))
        last = self.cfg.get("workspace", 0)
        self.show_workspace(last if last in (0, 1, 2) else 0, ask=False)
        try:
            QGuiApplication.styleHints().colorSchemeChanged.connect(lambda *_: self.set_theme(self.cfg.get("theme", "system"), save=False))
        except Exception:
            pass

    # ------------------------------------------------------------------ helpers
    def _pages(self):
        return (self.fast, self.prec, self.full)

    def _tools_status(self):
        st = engine.tool_status()
        ai = "AI denoiser ready" if st["deepfilter"] else "AI denoiser not found (built-in used)"
        au = "audio output ready" if self.session.player.available else "audio output unavailable"
        self.tool_label.setText(f"{ai}   |   ffmpeg {'ready' if st['ffmpeg'] else 'MISSING'}   |   {au}   ")

    def preset_dict(self, name, keep_output=True):
        if name.startswith("* "):
            d = dict(cfg.DEFAULTS)
            d.update(self.user_presets.get(name[2:], {}))
            return d
        d = cfg.preset_settings(name)
        if keep_output:
            explicit = set(cfg.PRESETS.get(name, {}))
            for k in OUT_KEYS:
                if k not in explicit:
                    d[k] = self.store.s[k]
        return d

    # ------------------------------------------------------------------ actions
    def _act(self, aid, text, fn, tip_text, checkable=False):
        a = QAction(text, self)
        a.setToolTip(tip_text)
        a.setStatusTip(tip_text)
        a.setCheckable(checkable)
        a.triggered.connect(lambda _=False, fn=fn: fn())
        self.acts[aid] = a
        self.addAction(a)
        return a

    def _make_actions(self):
        s = self.session
        A = self._act
        A("open", "Open", self.open_dialog, "Open an audio or video file.")
        A("export", "Export", self.export_dialog, "Process the whole file and save it.")
        A("auto", "Auto Clean", self.auto_clean, "Analyse the audio and clean it automatically - choose a preset or use our best guess.")
        A("play", "Play", s.toggle_play, "Play or pause.")
        A("stop", "Stop", s.stop, "Stop playback.")
        A("hear_a", "Hear original", lambda: s.set_ab("A"), "Hear the original recording.")
        A("hear_b", "Hear cleaned", lambda: s.set_ab("B"), "Hear the cleaned version.")
        A("undo", "Undo", lambda: self._route("undo"), "Undo the last edit.")
        A("redo", "Redo", lambda: self._route("redo"), "Redo.")
        A("ws1", "FastTrack", lambda: self.show_workspace(0), WS_TIPS["FastTrack"], True)
        A("ws2", "Precision", lambda: self.show_workspace(1), WS_TIPS["Precision"], True)
        A("ws3", "Full Studio", lambda: self.show_workspace(2), WS_TIPS["Full Studio"], True)
        A("batch", "Batch", lambda: self.show_workspace(3), "Batch: clean many files in one go.", True)
        A("guide", "Guide", lambda: self.show_workspace(4), "Open the quick guide.", True)
        A("trim", "Trim", lambda: self._route("trim"), "Keep only the selected region (waveform) or time range (timeline).")
        A("cut", "Cut", lambda: self._route("cut"), "Cut the selected region or clips.")
        A("silence", "Silence", lambda: self._route("silence"), "Mute the selected region or clips.")
        A("fade_in", "Fade in", lambda: self._route("fade_in"), "Fade in.")
        A("fade_out", "Fade out", lambda: self._route("fade_out"), "Fade out.")
        A("normalize", "Normalize", lambda: self._route("normalize"), "Set the loudest peak to -1 dB.")
        A("zoom_fit", "Zoom to fit", self.zoom_fit, "Show the whole file or project.")
        A("clip_copy", "Copy clips", lambda: self._tl_do("copy"), "Copy the selected clips.")
        A("clip_paste", "Paste clips", lambda: self._tl_do("paste"), "Paste clips at the playhead.")
        A("clip_clear", "Clear clips", lambda: self._tl_do("clear"), "Delete the selected clips.")
        A("clip_dup", "Duplicate clips", lambda: self._tl_do("duplicate"), "Duplicate the selected clips.")
        A("clip_repeat", "Repeat clips", lambda: self._tl_do("repeat"), "Repeat the selected clips.")
        A("sep", "Separate", lambda: self._tl_do("sep_playhead"), "Separate clips at the playhead.")
        A("nudge_l", "Nudge earlier", lambda: self._tl_do("do_nudge", -1), "Nudge selected clips earlier.")
        A("nudge_r", "Nudge later", lambda: self._tl_do("do_nudge", 1), "Nudge selected clips later.")
        A("save_proj", "Save project", self.save_project, "Save the multi-track project and copy its media into a project folder.")
        A("open_proj", "Open project", self.open_project, "Open a saved multi-track project.")
        A("stems", "Render stems", lambda: self._tl_do("render_stems"), "Export every track as its own WAV file.")
        A("theme", "Cycle theme", self.cycle_theme, "Switch between System, Dark and Light themes.")
        A("tut_basic", "Basic tutorial", lambda: self.start_tutorial("basic"), "Replay the basic tutorial.")
        A("tut_adv", "Advanced tutorial", lambda: self.start_tutorial("advanced"), "Replay the advanced tutorial.")
        self.ws_group = QActionGroup(self)
        for k in ("ws1", "ws2", "ws3", "batch", "guide"):
            self.ws_group.addAction(self.acts[k])

    def apply_shortcuts(self):
        ws = ["FastTrack", "Precision", "Full Studio"][min(self.stack.currentIndex(), 2)]
        saved = self.cfg.get("shortcuts", {}).get(ws, {})
        for aid, _, default in ACTIONS:
            if aid in self.acts:
                self.acts[aid].setShortcut(QKeySequence(saved.get(aid, default)))

    # ------------------------------------------------------------------ ribbon + menus
    def _ribbon(self):
        tb = QToolBar("Ribbon")
        tb.setObjectName("ribbon")
        tb.setMovable(False)
        tb.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)
        self.ribbon = tb
        logo = QLabel()
        pm = QPixmap(str(app_dir() / "assets" / "icon.png"))
        if not pm.isNull():
            logo.setPixmap(pm.scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo.setStyleSheet("background: transparent;")
        tip(logo, f"{APP_NAME} - built for M Hous3 Media.")
        tb.addWidget(logo)
        for k in ("open", "auto"):
            tb.addAction(self.acts[k])
        tb.addSeparator()
        for k in ("undo", "redo"):
            tb.addAction(self.acts[k])
        tb.addSeparator()
        tb.addAction(self.acts["export"])
        spacer = QWidget()
        spacer.setStyleSheet("background: transparent;")
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        for k in ("ws1", "ws2", "ws3"):
            tb.addAction(self.acts[k])
            tb.widgetForAction(self.acts[k]).setObjectName("wsbtn")
        tb.addSeparator()
        for k in ("batch", "guide"):
            tb.addAction(self.acts[k])
        tb.addSeparator()
        self.tut_btn = self._menu_button("Tutorial", "Replay the basic or advanced tutorial.", self._tutorial_menu())
        self.cust_btn = self._menu_button("Customize", "Change keyboard shortcuts and reset panel layouts.", self._customize_menu())
        self.theme_btn = self._menu_button("Theme", "Choose Light, Dark or System default.", self._theme_menu())
        self.macro_menu = QMenu(self)
        self.macro_btn = self._menu_button("Macros", "Run or create macros: your own sequences of editing commands.", self.macro_menu)
        self.refresh_macro_menu()
        for b in (self.macro_btn, self.tut_btn, self.cust_btn, self.theme_btn):
            tb.addWidget(b)
        tb.widgetForAction(self.acts["open"]).setObjectName("openbtn")
        tb.widgetForAction(self.acts["export"]).setObjectName("exportbtn")
        tb.widgetForAction(self.acts["auto"]).setObjectName("autobtn")

    def _menu_button(self, text, tip_text, menu):
        b = QToolButton()
        b.setText(text + "  v")
        b.setPopupMode(QToolButton.InstantPopup)
        b.setMenu(menu)
        tip(b, tip_text)
        return b

    def _tutorial_menu(self):
        from PySide6.QtWidgets import QMenu
        m = QMenu(self)
        m.addAction(self.acts["tut_basic"])
        m.addAction(self.acts["tut_adv"])
        m.addSeparator()
        a = QAction("Show tutorial when the app starts", self, checkable=True)
        a.setChecked(self.cfg.get("tutorial_on_start", True))
        a.toggled.connect(lambda v: self.cfg.set("tutorial_on_start", bool(v)))
        a.setToolTip("Turn the first-run tutorial on or off at startup.")
        m.addAction(a)
        return m

    def _customize_menu(self):
        from PySide6.QtWidgets import QMenu
        m = QMenu(self)
        a = QAction("Keyboard shortcuts...", self)
        a.setToolTip("Change the keyboard shortcuts for each workspace.")
        a.triggered.connect(lambda: ShortcutDialog(self).exec())
        m.addAction(a)
        ab = QAction("Auto Clean: skip the chooser and always use the recommendation", self, checkable=True)
        ab.setChecked(bool(self.cfg.get("always_best", False)))
        ab.setToolTip("When on, Auto Clean applies its recommended settings straight away. Turn off to be asked each time.")
        ab.toggled.connect(lambda v: self.cfg.set("always_best", bool(v)))
        m.addAction(ab)
        ss = QAction("Session settings...", self)
        ss.setToolTip("Ruler type (time, timecode, bars|beats), frame rate, tempo and playback buffer.")
        ss.triggered.connect(lambda: SessionDialog(self).exec())
        m.addAction(ss)
        mc = QAction("Edit macros...", self)
        mc.triggered.connect(lambda: MacroDialog(self).exec())
        m.addAction(mc)
        b = QAction("Reset this workspace's panel layout", self)
        b.triggered.connect(self.reset_layout)
        m.addAction(b)
        self.panels_menu = m.addMenu("Show / hide panels")
        self.panels_menu.aboutToShow.connect(self._fill_panels)
        return m

    def _theme_menu(self):
        from PySide6.QtWidgets import QMenu
        m = QMenu(self)
        g = QActionGroup(self)
        self.theme_acts = {}
        for mode, label in (("system", "System default"), ("dark", "Dark"), ("light", "Light")):
            a = QAction(label, self, checkable=True)
            a.setActionGroup(g)
            a.triggered.connect(lambda _=False, mode=mode: self.set_theme(mode))
            m.addAction(a)
            self.theme_acts[mode] = a
        self.theme_acts[self.cfg.get("theme", "system")].setChecked(True)
        return m

    def _menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        for k in ("open", "export"):
            f.addAction(self.acts[k])
        f.addSeparator()
        for k in ("open_proj", "save_proj", "stems"):
            f.addAction(self.acts[k])
        tm = f.addMenu("New project from template")
        for name in fx.SESSION_TEMPLATES:
            tm.addAction(name, lambda name=name: self.new_from_template(name))
        ex = f.addMenu("Export session as")
        for label, kind in (("REAPER project (.rpp)", "rpp"), ("AAF (.aaf)", "aaf"), ("CMX3600 EDL (one per track)", "edl"), ("Final Cut Pro XML (.xml)", "fcpxml")):
            ex.addAction(label, lambda kind=kind: self.export_session(kind))
        im = f.addMenu("Import session from")
        for label, kind in (("REAPER project (.rpp)", "rpp"), ("Final Cut Pro XML (.xml)", "fcpxml")):
            im.addAction(label, lambda kind=kind: self.import_session(kind))
        f.addSeparator()
        q = QAction("Quit", self)
        q.setShortcut("Ctrl+Q")
        q.triggered.connect(self.close)
        f.addAction(q)
        e = mb.addMenu("&Edit")
        for k in ("undo", "redo"):
            e.addAction(self.acts[k])
        e.addSeparator()
        for k in ("trim", "cut", "silence", "fade_in", "fade_out", "normalize"):
            e.addAction(self.acts[k])
        e.addSeparator()
        for k in ("clip_copy", "clip_paste", "clip_clear", "clip_dup", "clip_repeat", "sep", "nudge_l", "nudge_r"):
            e.addAction(self.acts[k])
        v = mb.addMenu("&View")
        v.addAction(self.acts["zoom_fit"])
        tm = v.addMenu("Theme")
        for a in self.theme_acts.values():
            tm.addAction(a)
        v.addSeparator()
        r = QAction("Reset panel layout", self)
        r.triggered.connect(self.reset_layout)
        v.addAction(r)
        w = mb.addMenu("&Workspace")
        for k in ("ws1", "ws2", "ws3", "batch"):
            w.addAction(self.acts[k])
        c = mb.addMenu("&Customize")
        ks = QAction("Keyboard shortcuts...", self)
        ks.triggered.connect(lambda: ShortcutDialog(self).exec())
        c.addAction(ks)
        t = mb.addMenu("&Tutorial")
        t.addAction(self.acts["tut_basic"])
        t.addAction(self.acts["tut_adv"])
        h = mb.addMenu("&Help")
        h.addAction(self.acts["guide"])
        ab = QAction("About", self)
        ab.triggered.connect(lambda: QMessageBox.about(
            self, APP_NAME, f"{APP_NAME} {APP_VERSION}\n\nBuilt for M Hous3 Media.\n(c) M Hous3 Media. All rights reserved.\n\n"
                            "Free, offline audio cleanup and cinema finishing.\nPowered by ffmpeg and DeepFilterNet."))
        h.addAction(ab)

    def _fill_panels(self):
        self.panels_menu.clear()
        pg = self.stack.currentWidget()
        if hasattr(pg, "docks"):
            for d in pg.docks.values():
                self.panels_menu.addAction(d.toggleViewAction())

    # ------------------------------------------------------------------ theme / layout
    def set_theme(self, mode, save=True):
        theme.apply(QApplication.instance(), mode)
        if save:
            self.cfg.set("theme", mode)
        if mode in self.theme_acts:
            self.theme_acts[mode].setChecked(True)
        self.guide.refresh()
        self.tutorial._style()
        self.update()

    def cycle_theme(self):
        order = ["system", "dark", "light"]
        cur = self.cfg.get("theme", "system")
        self.set_theme(order[(order.index(cur) + 1) % 3])

    def reset_layout(self):
        pg = self.stack.currentWidget()
        if hasattr(pg, "reset_layout"):
            pg.reset_layout()

    def zoom_fit(self):
        tl = self._tl()
        if tl is not None:
            tl.view.zoom_fit()
            return
        pg = self.stack.currentWidget()
        if hasattr(pg, "wave"):
            pg.wave.view.zoom_fit()

    # ------------------------------------------------------------------ workspaces
    def _confirm_keep(self):
        s = self.session
        if not (s.audio is not None and s.cleaned_by and not s.popup_done):
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Keep your changes?")
        box.setIcon(QMessageBox.Question)
        box.setText(f"This file was cleaned with {s.cleaned_by}.\n\nDo you want to keep the current changes and keep refining "
                    "them here, or start again from the original recording?")
        keep = box.addButton("Keep my changes and refine", QMessageBox.AcceptRole)
        orig = box.addButton("Start from the original", QMessageBox.DestructiveRole)
        cancel = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(keep)
        box.exec()
        b = box.clickedButton()
        if b is cancel or b is None:
            return False
        s.popup_done = True
        if b is orig:
            s.reset_to_original()
        return True

    def show_workspace(self, i, ask=True):
        cur = self.stack.currentIndex()
        if i != cur and ask and cur in (0, 1, 2) and i in (0, 1, 2) and not self._confirm_keep():
            self.ws_group.blockSignals(True)
            self.acts[["ws1", "ws2", "ws3", "batch", "guide"][cur]].setChecked(True)
            self.ws_group.blockSignals(False)
            return
        self.stack.setCurrentIndex(i)
        self.acts[["ws1", "ws2", "ws3", "batch", "guide"][i]].setChecked(True)
        if i in (0, 1, 2):
            self.cfg.set("workspace", i)
        self.apply_shortcuts()
        pg = self.stack.currentWidget()
        if hasattr(pg, "on_show"):
            pg.on_show()
        names = ["FastTrack", "Precision", "Full Studio", "Batch", "Guide"]
        self.statusBar().showMessage(f"Workspace: {names[i]}", 2500)
        self.setWindowTitle(APP_NAME + (f" - {self.project.name}" if self.session.project_mode else ""))

    # ------------------------------------------------------------------ multi-track routing
    def _tl(self):
        pg = self.stack.currentWidget()
        if isinstance(pg, PrecisionPage) and pg.tabs.currentIndex() == 0:
            return pg.timeline
        return None

    def _tl_do(self, name, *args):
        tl = self._tl()
        if tl is None:
            self.statusBar().showMessage("Open the Multi-track tab in Precision or Full Studio to edit clips.", 4000)
            return
        getattr(tl, name)(*args)

    def _route(self, what):
        s, tl = self.session, self._tl()
        if tl is None:
            {"undo": s.do_undo, "redo": s.do_redo, "trim": s.trim, "cut": s.cut, "silence": s.silence,
             "fade_in": lambda: s.fade(True), "fade_out": lambda: s.fade(False), "normalize": s.normalize}[what]()
            return
        {"undo": self.project.undo, "redo": self.project.redo, "trim": tl.trim_range, "cut": tl.cut, "silence": tl.clip_mute,
         "fade_in": lambda: tl.fade_playhead(True), "fade_out": lambda: tl.fade_playhead(False), "normalize": tl.clip_normalize}[what]()

    def refresh_macro_menu(self):
        m = self.macro_menu
        m.clear()
        for name, steps in self.cfg.get("macros", {}).items():
            m.addAction(name, lambda steps=steps, name=name: self.statusBar().showMessage(f"Macro '{name}': {run_macro(self, steps)} step(s) done.", 4000))
        if m.actions():
            m.addSeparator()
        m.addAction("Edit macros...", lambda: MacroDialog(self).exec())

    def new_from_template(self, name):
        self.project.new_from_template(name)
        self.attach_project()
        self.show_workspace(1, ask=False)
        self.prec.tabs.setCurrentIndex(1)
        self.statusBar().showMessage(f"New project from template: {name}. Import audio onto its tracks.", 6000)

    def export_session(self, kind, path=None):
        if not self.project.tracks:
            QMessageBox.information(self, APP_NAME, "The project is empty. Import audio on the Multi-track tab first.")
            return
        fps = self.project.settings.get("fps", "25")
        try:
            if kind == "edl":
                path = path or QFileDialog.getExistingDirectory(self, "Choose a folder for the EDL files")
                if not path:
                    return
                out = ix.export_edl(self.project, path, fps)
                msg = f"Saved {len(out)} EDL file(s) in {path}"
            else:
                ext, filt, fn = {"rpp": (".rpp", "REAPER project (*.rpp)", ix.export_rpp), "aaf": (".aaf", "AAF (*.aaf)", ix.export_aaf),
                                 "fcpxml": (".xml", "Final Cut Pro XML (*.xml)", lambda p, d: ix.export_fcpxml(p, d, fps))}[kind]
                path = path or QFileDialog.getSaveFileName(self, "Export session", str(Path.home() / f"{self.project.name}{ext}"), filt)[0]
                if not path:
                    return
                fn(self.project, path) if kind != "fcpxml" else fn(self.project, path)
                msg = f"Exported {Path(path).name} (audio copied into {Path(path).stem}_media)" if kind != "aaf" else f"Exported {Path(path).name} with the audio embedded"
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Export failed:\n{e}")
            return
        self.statusBar().showMessage(msg + ". Inserts, sends and automation are not carried across.", 8000)

    def import_session(self, kind, path=None):
        filt = {"rpp": "REAPER project (*.rpp)", "fcpxml": "Final Cut Pro XML (*.xml)"}[kind]
        path = path or QFileDialog.getOpenFileName(self, "Import session", "", filt)[0]
        if not path:
            return
        try:
            (ix.import_rpp if kind == "rpp" else ix.import_fcpxml)(self.project, path)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Import failed:\n{e}")
            return
        self.attach_project()
        self.show_workspace(1, ask=False)
        self.prec.tabs.setCurrentIndex(0)
        self.prec.timeline.view.zoom_fit()
        self.statusBar().showMessage(f"Imported {Path(path).name}: {len(self.project.tracks)} track(s).", 6000)

    def attach_project(self):
        self.session.attach_project(self.project)

    def save_project(self):
        if not self.project.tracks:
            QMessageBox.information(self, APP_NAME, "The project is empty. Import some audio on the Multi-track tab first.")
            return
        path = self.project.path
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save project", str(Path.home() / "My Project.mh3p"), "MHous3 project (*.mh3p)")
            if not path:
                return
        try:
            out = self.project.save(path)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Could not save the project:\n{e}")
            return
        self.statusBar().showMessage(f"Saved {out.name} and copied its media into {out.stem}_media", 5000)

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "MHous3 project (*.mh3p)")
        if not path:
            return
        try:
            self.project.load(path)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Could not open the project:\n{e}")
            return
        self.attach_project()
        self.show_workspace(1, ask=False)
        self.prec.tabs.setCurrentIndex(0)
        self.prec.timeline.view.zoom_fit()
        self.statusBar().showMessage(f"Opened {Path(path).name}", 4000)

    # ------------------------------------------------------------------ file + cleanup actions
    def open_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open audio or video", "", MEDIA_FILTER)
        if p:
            self.session.load(p)

    def export_dialog(self):
        s = self.session
        if s.audio is None:
            QMessageBox.information(self, APP_NAME, "Open a file first.")
            return
        st = dict(self.store.s)
        ext = cfg.FORMATS[st["out_format"]][0]
        if ext in (".mp4", ".mkv") and Path(s.src_path).suffix.lower() not in cfg.VIDEO_EXT:
            ext, st["out_format"] = ".wav", "WAV"
        default = str(engine.default_output_path(s.src_path, st, None, "_MHous3"))
        dst, _ = QFileDialog.getSaveFileName(self, "Export", default, f"{ext.upper()[1:]} (*{ext})")
        if dst:
            s.export_to(dst, st)

    def ask_gain(self):
        if self.session.audio is None:
            return
        db, ok = QInputDialog.getDouble(self, "Gain", "Change volume by (dB):", 3.0, -48, 48, 1)
        if ok:
            self.session.gain(db)

    def pick_preset(self, name):
        if name == PresetCards.AUTO:
            return self.auto_clean()
        self.session.apply_settings(self.preset_dict(name), name)

    def auto_clean(self):
        s = self.session
        if s.audio is None:
            if QMessageBox.question(self, APP_NAME, "Open a file first. Open one now?") == QMessageBox.Yes:
                self.open_dialog()
            return
        self.statusBar().showMessage("Listening to your audio...")
        mono, keep = s.mono_a, dict(self.store.s)

        def work(prog, cancel):
            f = analyzer.analyze(mono)
            return (f, *analyzer.recommend(f, keep))

        t = Task(work)
        s._hold(t)
        t.sig.done.connect(self._auto_ready)
        t.sig.failed.connect(lambda m: QMessageBox.warning(self, APP_NAME, f"Auto Clean could not analyse the audio:\n{m}"))
        s.pool.start(t)

    def _auto_ready(self, res):
        feats, settings, notes, scenario = res
        self.statusBar().showMessage("Ready")
        if self.cfg.get("always_best", False):
            self.session.apply_settings(settings, PresetCards.AUTO)
            self.statusBar().showMessage(f"Auto Clean applied: {scenario}", 4000)
            return
        dlg = AutoCleanDialog(self, scenario, notes, self.user_presets)
        if dlg.exec() != QDialog.Accepted:
            return
        if dlg.always.isChecked() and dlg.choice == "auto":
            self.cfg.set("always_best", True)
        if dlg.choice == "auto":
            self.session.apply_settings(settings, PresetCards.AUTO)
        else:
            self.pick_preset(dlg.choice)

    # ------------------------------------------------------------------ tutorials
    def _ribbon_widget(self, key):
        return self.ribbon.widgetForAction(self.acts[key])

    def _tutorial_steps(self, kind):
        f, p, fu = self.fast, self.prec, self.full
        go = lambda i: (lambda: self.show_workspace(i, ask=False))
        if kind == "basic":
            return [
                dict(title="Welcome to MHous3 Audio Studio", text="Built for M Hous3 Media. Let's clean up your first recording in a few quick steps. You can skip at any time.", pre=go(0)),
                dict(title="Your workspaces", text="FastTrack is for quick, great results. Precision gives you manual control. Full Studio is the professional environment. Hover over each button to see what it does.",
                     target=lambda: self._ws_box()),
                dict(title="Open a recording", text="Click Open to choose an audio or video file, or just drag one onto the window.", target=lambda: self._ribbon_widget("open")),
                dict(title="Live spectrogram", text="This shows the sound you are hearing as it plays. Low sounds are at the bottom, high sounds at the top. The waveform below shows loudness over time.",
                     target=lambda: f.spec),
                dict(title="Auto Clean", text="The easiest way to start. We listen to your file, then you either pick a preset or use our recommended settings.", target=lambda: f.auto_btn),
                dict(title="Quick controls", text="Big knobs for the things people want most: Volume Boost, Noise Reduction, Echo Removal, Room Expansion and Reverb. Drag a knob up or down. Double-click to reset it.",
                     target=lambda: f.knobs[0].parentWidget()),
                dict(title="Presets", text="Ready-made looks for common situations. Click a card to apply it. The first card is Auto Clean.", target=lambda: f.cards),
                dict(title="Compare before and after", text="Press Play, then switch between A (original) and B (cleaned) at any time. Match loudness keeps the comparison fair.", target=lambda: f.transport),
                dict(title="Export", text="When you are happy, click Export to save your cleaned file. For many files, use the Batch button.", target=lambda: self._ribbon_widget("export")),
            ]
        return [
            dict(title="Precision workspace", text="Precision is for focused manual control. Panels (docks) can be dragged, floated, tabbed or hidden - arrange it your way.", pre=go(1),
                 target=lambda: self._ws_box()),
            dict(title="Multi-track timeline", text="Your file is Track 1. Drop more files here to add tracks. Drag clips to move them, drag their edges to trim, drag the top corners for fades and the dashed line for clip gain. Press A on a track to draw volume automation.",
                 pre=lambda: p.tabs.setCurrentIndex(0), target=lambda: p.timeline.view),
            dict(title="Editing tools", pre=lambda: p.tabs.setCurrentIndex(2), text="In the Waveform editor tab, select a region by dragging on the waveform, then use Trim, Cut, Silence, Fades, Gain, Normalize or Reverse. Every edit can be undone.", target=lambda: p.tools),
            dict(title="Waveform and selection", text="Scroll to zoom, Shift+scroll to move, drag to select, double-click to select everything. The amber shape is your cleaned result over the grey original.", target=lambda: p.wave),
            dict(title="Mixer", text="The console: a strip for every track, bus and VCA with faders, pan, sends, insert effects and routing. Double-click an effect to edit it, or add your own JSFX script.",
                 pre=lambda: p.tabs.setCurrentIndex(1), target=lambda: p.mixer),
            dict(title="Pro Controls", text="Every parameter is here: six-band EQ, gate, de-esser, compressor, saturation, width, reverb details, echo decay and the Dolby 5.1 mix.", pre=lambda: (p.tabs.setCurrentIndex(0), p.docks["Pro Controls"].raise_()),
                 target=lambda: p.docks["Pro Controls"]),
            dict(title="Refining an automatic clean", text="After Auto Clean or a preset, you are asked whether to keep your changes or start again from the original when you change workspace.", target=None),
            dict(title="Full Studio", text="The professional environment: the multi-track timeline, the mixer, 33 effects and JSFX scripting, loudness meters and session export. Live automation recording, MIDI and spatial tools are still to come.", pre=go(2),
                 target=lambda: self._ws_box()),
            dict(title="Batch", text="Clean dozens of files with the same settings, saved next to each original or into one folder.", pre=go(3), target=lambda: self._ribbon_widget("batch")),
            dict(title="Make it yours", text="Customize changes keyboard shortcuts separately for each workspace and lets you show or hide panels. Drag any panel title to rearrange the layout.", pre=go(1),
                 target=lambda: self.cust_btn),
            dict(title="Themes", text="Choose Dark, Light, or follow your computer (System default).", target=lambda: self.theme_btn),
        ]

    def _ws_box(self):
        return self._make_ws_proxy()

    def start_tutorial(self, kind="basic"):
        self.tutorial._style()
        self.tutorial.kind = kind
        self.tutorial.start(self._tutorial_steps(kind), "Tutorial")

    def _make_ws_proxy(self):
        """A transparent widget spanning the three workspace buttons (used as the spotlight target)."""
        a, c = self.ribbon.widgetForAction(self.acts["ws1"]), self.ribbon.widgetForAction(self.acts["ws3"])
        proxy = getattr(self, "_proxy", None) or QWidget(self.ribbon)
        proxy.setAttribute(Qt.WA_TransparentForMouseEvents)
        proxy.setStyleSheet("background: transparent;")
        proxy.setGeometry(a.geometry().united(c.geometry()))
        proxy.show()
        proxy.lower()
        self._proxy = proxy
        return proxy

    def _tutorial_done(self, completed):
        self.cfg.set("tutorial_done", True)
        self.show_workspace(0, ask=False)

    def after_show(self):
        if self.cfg.get("tutorial_on_start", True) and not self.cfg.get("tutorial_done", False):
            QTimer.singleShot(700, lambda: self.start_tutorial("basic"))

    # ------------------------------------------------------------------ drag and drop / close
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return
        if self.stack.currentIndex() == 3 or len(paths) > 1 or Path(paths[0]).is_dir():
            self.show_workspace(3, ask=False)
            self.batch.add_paths(paths)
        else:
            self.session.load(paths[0])

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if getattr(self, "tutorial", None) is not None and self.tutorial.isVisible():
            self.tutorial.setGeometry(self.rect())

    def closeEvent(self, e):
        for pg in self._pages():
            self.cfg.set(f"layout_{pg.NAME}", pg.save_layout(), save=False)
        self.cfg.save()
        self.session.shutdown()
        super().closeEvent(e)
