"""Custom macros: a named list of commands that runs in order (stored in the config file)."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget, QMessageBox, QPushButton, QVBoxLayout)

from .widgets import tip

# id: (label, target, method, args)  -  "tl" = the multi-track editor, "main" = the main window
COMMANDS = {
    "select_all": ("Select all clips", "tl", "select_all", ()),
    "sep_transients": ("Separate at transients", "tl", "sep_transients", ()),
    "sep_grid": ("Separate at grid", "tl", "sep_grid", ()),
    "sep_playhead": ("Separate at playhead", "tl", "sep_playhead", ()),
    "fade5": ("Fades 5 ms on all selected clips", "tl", "fade_all", (5,)),
    "fade20": ("Fades 20 ms on all selected clips", "tl", "fade_all", (20,)),
    "fade100": ("Fades 100 ms on all selected clips", "tl", "fade_all", (100,)),
    "normalize_clips": ("Normalize selected clips", "tl", "clip_normalize", ()),
    "mute_clips": ("Mute / unmute selected clips", "tl", "clip_mute", ()),
    "lock_clips": ("Lock / unlock selected clips", "tl", "clip_lock", ()),
    "duplicate": ("Duplicate selected clips", "tl", "duplicate", ()),
    "copy": ("Copy selected clips", "tl", "copy", ()),
    "paste": ("Paste at playhead", "tl", "paste", ()),
    "clear": ("Clear selected clips", "tl", "clear", ()),
    "add_track": ("Add track", "tl", "add_track", ()),
    "add_bus": ("Add bus", "tl", "add_bus", ()),
    "zoom_fit": ("Zoom to fit", "tl", "zoom_fit_view", ()),
    "auto_clean": ("Auto Clean the mix", "main", "auto_clean", ()),
    "preset_cinema": ("Apply preset: Cinema Dialogue", "main", "pick_preset", ("Cinema Dialogue (Hollywood)",)),
    "preset_podcast": ("Apply preset: Podcast / Voice-over", "main", "pick_preset", ("Podcast / Voice-over",)),
    "stems": ("Render stems", "tl", "render_stems", ()),
    "save_project": ("Save project", "main", "save_project", ()),
}


def run_macro(main, steps):
    """Run command ids in order.  Returns the number of steps executed."""
    done = 0
    for cid in steps:
        if cid not in COMMANDS:
            continue
        label, target, method, args = COMMANDS[cid]
        obj = main._tl() if target == "tl" else main
        if obj is None:
            main.statusBar().showMessage("Macros that edit clips need the Multi-track tab in Precision or Full Studio.", 5000)
            break
        getattr(obj, method)(*args)
        done += 1
    return done


class MacroDialog(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Macros")
        self.setMinimumSize(640, 440)
        self.macros = dict(main.cfg.get("macros", {}))
        v = QVBoxLayout(self)
        v.addWidget(QLabel("A macro runs several commands in order. Build one, then run it from the Macros menu."))
        row = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        row.addLayout(left, 1)
        row.addLayout(right, 2)
        v.addLayout(row, 1)
        self.names = QListWidget()
        self.names.currentRowChanged.connect(self._show)
        left.addWidget(QLabel("Your macros"))
        left.addWidget(self.names, 1)
        for txt, fn in (("New...", self._new), ("Delete", self._delete)):
            b = QPushButton(txt)
            b.clicked.connect(lambda _=False, fn=fn: fn())
            left.addWidget(b)
        right.addWidget(QLabel("Steps"))
        self.steps = QListWidget()
        right.addWidget(self.steps, 1)
        self.cmd = QComboBox()
        for cid, (label, *_r) in COMMANDS.items():
            self.cmd.addItem(label, cid)
        tip(self.cmd, "Choose a command to add to the macro.")
        right.addWidget(self.cmd)
        r2 = QHBoxLayout()
        for txt, fn, tt in (("Add step", self._add, "Add the chosen command."), ("Remove", self._rem, "Remove the selected step."),
                            ("Up", lambda: self._mv(-1), "Move the step earlier."), ("Down", lambda: self._mv(1), "Move the step later.")):
            b = QPushButton(txt)
            b.clicked.connect(lambda _=False, fn=fn: fn())
            tip(b, tt)
            r2.addWidget(b)
        right.addLayout(r2)
        bottom = QHBoxLayout()
        run = QPushButton("Run now")
        run.clicked.connect(self._run)
        tip(run, "Run the selected macro on the current project.")
        ok = QPushButton("Save and close")
        ok.setObjectName("primary")
        ok.clicked.connect(self._save)
        bottom.addWidget(run)
        bottom.addStretch(1)
        bottom.addWidget(ok)
        v.addLayout(bottom)
        self._fill()

    def _fill(self, select=None):
        self.names.clear()
        self.names.addItems(list(self.macros))
        if self.macros:
            self.names.setCurrentRow(list(self.macros).index(select) if select in self.macros else 0)

    def _cur(self):
        it = self.names.currentItem()
        return it.text() if it else None

    def _show(self, _row):
        self.steps.clear()
        n = self._cur()
        for cid in self.macros.get(n, []):
            self.steps.addItem(COMMANDS[cid][0] if cid in COMMANDS else cid)

    def _new(self):
        name, ok = QInputDialog.getText(self, "New macro", "Macro name:")
        if ok and name.strip():
            self.macros.setdefault(name.strip(), [])
            self._fill(name.strip())

    def _delete(self):
        n = self._cur()
        if n:
            del self.macros[n]
            self._fill()

    def _add(self):
        n = self._cur()
        if n is not None:
            self.macros[n].append(self.cmd.currentData())
            self._show(0)

    def _rem(self):
        n, i = self._cur(), self.steps.currentRow()
        if n and 0 <= i < len(self.macros[n]):
            del self.macros[n][i]
            self._show(0)

    def _mv(self, d):
        n, i = self._cur(), self.steps.currentRow()
        if n and 0 <= i < len(self.macros[n]) and 0 <= i + d < len(self.macros[n]):
            lst = self.macros[n]
            lst[i], lst[i + d] = lst[i + d], lst[i]
            self._show(0)
            self.steps.setCurrentRow(i + d)

    def _run(self):
        n = self._cur()
        if n:
            done = run_macro(self.main, self.macros[n])
            self.main.statusBar().showMessage(f"Macro '{n}': {done} step(s) done.", 4000)

    def _save(self):
        self.main.cfg.set("macros", self.macros)
        self.main.refresh_macro_menu()
        self.accept()
