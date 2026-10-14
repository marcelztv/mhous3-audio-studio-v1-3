"""Everything about the file being worked on (audio, edits, previews, playback, export) - no widgets.
All three workspaces are views of one Session, so nothing is lost when switching workspace."""
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, QTimer, Signal

from . import dsp, engine, settings as cfg
from .dsp import SR
from .player import ABPlayer


class _Sig(QObject):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)


class Task(QRunnable):
    """Runs fn(progress, cancel) on a low-priority worker thread so the interface never stalls."""

    def __init__(self, fn):
        super().__init__()
        self.setAutoDelete(False)
        self.fn, self.sig, self.cancel = fn, _Sig(), threading.Event()

    def run(self):
        try:
            QThread.currentThread().setPriority(QThread.LowPriority)
        except Exception:
            pass
        msg, res = None, None
        try:
            res = self.fn(self.sig.progress.emit, self.cancel)
        except engine.Cancelled:
            msg = "Cancelled"
        except Exception as e:
            msg = str(e) or type(e).__name__
        try:
            (self.sig.failed.emit(msg) if msg is not None else self.sig.done.emit(res))
        except RuntimeError:
            pass


class Session(QObject):
    audioChanged = Signal()
    previewStarted = Signal()
    previewReady = Signal()
    previewFailed = Signal(str)
    selectionChanged = Signal()
    tick = Signal()
    progress = Signal(float, str)
    info = Signal(str)
    stateChanged = Signal()
    busy = Signal(str)
    idle = Signal()
    error = Signal(str)
    loaded = Signal(str)
    exported = Signal(str, object)

    def __init__(self, store):
        super().__init__()
        self.store = store
        self.audio = self.original = self.mono_a = self.mono_b = None
        self.src_path = None
        self.undo, self.redo = [], []
        self.player = ABPlayer(SR)
        self.p_off = 0
        self.preview = None
        self.task = None
        self._pending = False
        self._seek_after = None
        self._want_b = False
        self.sel = None
        self.auto_preview = True
        self.scope = 0
        self.match = True
        self.loop = False
        self.cache = {}
        self.token = 0
        self.project = None
        self.project_mode = False
        self._prev_auto = True
        self.remix_timer = QTimer(self)
        self.remix_timer.setSingleShot(True)
        self.remix_timer.timeout.connect(self._remix)
        self.cleaned_by = None      # "Auto Clean" / preset name once automatic cleanup was applied
        self.popup_done = False
        self.pool = QThreadPool.globalInstance()
        self._tasks = []
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.timeout.connect(self.request_preview)
        self.ticker = QTimer(self)
        self.ticker.timeout.connect(self.tick.emit)
        self.ticker.start(33)
        store.changed.connect(self._settings_changed)

    # ------------------------------------------------------------ helpers
    def _hold(self, t):
        self._tasks = [x for x in self._tasks if x is not t][-6:] + [t]

    @property
    def n(self):
        return 0 if self.audio is None else len(self.audio)

    def global_pos(self):
        return self.p_off + self.player.pos

    def spec_source(self):
        """(mono array, offset) of whatever is currently audible - used by the live spectrogram."""
        if self.player.src == "B" and self.mono_b is not None:
            return self.mono_b, self.p_off
        return self.mono_a, self.p_off

    # ------------------------------------------------------------ loading
    def load(self, path):
        self.busy.emit(f"Reading {Path(path).name}...")
        t = Task(lambda prog, cancel: engine.decode(path, cancel))
        self._hold(t)
        t.sig.done.connect(lambda a: self._loaded(path, a))
        t.sig.failed.connect(lambda m: (self.idle.emit(), self.error.emit(f"Could not open file:\n{m}")))
        self.pool.start(t)

    def _loaded(self, path, a):
        self.idle.emit()
        self.stop()
        if self.project_mode:
            self.project_mode = False
            self.auto_preview = self._prev_auto
        self.src_path, self.audio, self.original = path, a, a.copy()
        self.mono_a, self.mono_b = a.mean(axis=1).astype(np.float32), None
        self.undo, self.redo, self.preview, self.p_off, self.sel = [], [], None, 0, None
        self.token += 1
        self.cache.clear()
        self.cleaned_by, self.popup_done = None, False
        self.player.set_audio(a, None)
        self.player.seek(0)
        self.set_ab("A")
        self._want_b = True
        self.audioChanged.emit()
        self.stateChanged.emit()
        self.loaded.emit(path)
        self.info.emit("Loaded. Click Auto Clean, pick a preset, or turn the knobs - the preview updates automatically.")
        if self.auto_preview:
            self.request_preview()

    # ------------------------------------------------------------ multi-track project mode
    def attach_project(self, project):
        """Make the timeline mixdown the working audio.  The usual cleanup chain then runs on the mix (A/B still works)."""
        if self.project is None:
            self.project = project
            project.add_listener(self._schedule_remix)
        if not self.project_mode:
            self._prev_auto = self.auto_preview
            self.auto_preview = False
            self.project_mode = True
            self.stateChanged.emit()
        if not self.src_path or Path(self.src_path).name != f"{project.name}.wav":
            self.src_path = str(Path.home() / f"{project.name}.wav")
        self._schedule_remix()

    def _schedule_remix(self):
        if self.project_mode:
            self.remix_timer.start(120)

    def _remix(self):
        if not (self.project_mode and self.project):
            return
        mix = self.project.mixdown()
        if len(mix) == 0:
            return
        self.set_array(mix)

    def set_array(self, a):
        """Replace the working audio with an in-memory array (used by the timeline). Keeps view and position."""
        keep = self.global_pos()
        self.audio = np.ascontiguousarray(a, dtype=np.float32)
        self.mono_a, self.mono_b = self.audio.mean(axis=1).astype(np.float32), None
        if self.original is None:
            self.original = self.audio.copy()
        self.preview, self.p_off, self.sel = None, 0, None
        self.token += 1
        self.cache.clear()
        was_playing = self.player.playing
        self.player.set_audio(self.audio, None)
        self.player.seek(min(keep, len(self.audio)))
        self.player.playing = was_playing and self.player.available
        self.set_ab("A")
        self._want_b = True
        self.audioChanged.emit()
        if self.auto_preview:
            self.debounce.start(300)

    # ------------------------------------------------------------ preview
    def _settings_changed(self, key):
        if key == "out_format" or self.audio is None:
            return
        if self.auto_preview:
            self.debounce.start(600)

    def preview_range(self):
        n = len(self.audio)
        if self.sel and self.sel[1] - self.sel[0] > SR // 4:
            return int(self.sel[0]), int(self.sel[1])
        return (0, n)

    def request_preview(self):
        if self.audio is None:
            return
        if self.task is not None:
            self._pending = True
            self.task.cancel.set()
            return
        s, e = self.preview_range()
        seg, sett, tok = self.audio[s:e].copy(), dict(self.store.s), (self.token, s, e)
        t = self.task = Task(lambda prog, cancel: (*engine.render(seg, sett, prog, cancel, self.cache, tok), s, e, seg))
        t.sig.progress.connect(self.progress.emit)
        t.sig.done.connect(self._preview_done)
        t.sig.failed.connect(self._preview_failed)
        self.previewStarted.emit()
        self._hold(t)
        self.pool.start(t)

    def _preview_failed(self, msg):
        self.task = None
        self.previewFailed.emit(msg)
        if msg != "Cancelled":
            self.info.emit(f"Preview failed: {msg}")
        if self._pending:
            self._pending = False
            self.request_preview()

    def _preview_done(self, res):
        y, info, s, e, seg = res
        self.task = None
        if self._pending:
            self._pending = False
            self.request_preview()
            return
        gpos = self.global_pos()
        was_playing = self.player.playing
        self.preview = (y, info, s, e)
        b = dsp.downmix_to_stereo(y)
        self.mono_b = b.mean(axis=1).astype(np.float32)
        self.player.set_audio(seg, b)
        self.p_off = s
        self.player.seek(gpos - s if s <= gpos <= e else 0)
        self.player.playing = was_playing and self.player.available
        self.apply_loop()
        self.apply_match()
        self.set_ab("B" if (self._want_b or self.player.src == "B") else "A")
        self._want_b = False
        txt = (f"Loudness  {info['lufs_in']:.1f} -> {info['lufs_out']:.1f} LUFS   |   Peak {info['peak_in_db']:.1f} -> "
               f"{info['peak_out_db']:.1f} dBFS   |   {info['channels']} ch")
        if "declip_runs" in info:
            txt += f"   |   Clipping repaired: {info['declip_runs']} spots ({info['clipped_fraction']:.2%})"
        if "nr_engine" in info:
            txt += f"   |   Noise: {info['nr_engine']}"
        if "warning" in info:
            txt += f"\nWarning: {info['warning']}"
        if (s, e) != (0, len(self.audio)):
            txt += f"\nPreviewing {s / SR:.1f}-{e / SR:.1f} s. Export always processes the whole file."
        self.info.emit(txt)
        self.previewReady.emit()
        if self._seek_after is not None:
            self.seek_global(self._seek_after)
            self._seek_after = None

    # ------------------------------------------------------------ transport
    def toggle_play(self):
        if self.audio is None:
            return
        if self.player.playing:
            self.player.pause()
        else:
            try:
                self.player.play()
            except Exception as e:
                self.error.emit(f"Audio playback problem:\n{e}")

    def stop(self):
        self.player.stop()

    def set_ab(self, which):
        if which == "B" and self.player.b is None:
            which = "A"
        self.player.switch(which)
        self.stateChanged.emit()

    def apply_match(self):
        if self.preview is None:
            self.player.gain_a = 1.0
            return
        info = self.preview[1]
        g = 1.0
        if self.match and np.isfinite(info["lufs_in"]) and np.isfinite(info["lufs_out"]):
            g = float(dsp.db2lin(np.clip(info["lufs_out"] - info["lufs_in"], -30, 30)))
            pk = float(np.max(np.abs(self.player.a))) if self.player.a is not None and self.player.a.size else 1.0
            g = min(g, 0.98 / max(pk, 1e-6))
        self.player.gain_a = g

    def set_match(self, v):
        self.match = bool(v)
        self.apply_match()

    def apply_loop(self):
        if self.loop and self.sel:
            a, b = self.sel[0] - self.p_off, self.sel[1] - self.p_off
            self.player.loop = (max(0, a), min(self.player.length(), b)) if b > 0 else None
        else:
            self.player.loop = None

    def set_loop(self, v):
        self.loop = bool(v)
        self.apply_loop()

    def seek_global(self, g):
        if self.audio is None:
            return
        n = self.player.length()
        if self.preview is not None and not (self.p_off <= g <= self.p_off + n):
            self._seek_after = g
            self.request_preview()
            return
        self.player.seek(g - self.p_off)

    def set_selection(self, sel):
        sel = (int(sel[0]), int(sel[1])) if sel else None
        if sel == self.sel:
            return
        self.sel = sel
        self.apply_loop()
        self.selectionChanged.emit()
        if self.auto_preview and self.audio is not None and sel:
            self.debounce.start(900)

    def set_auto_preview(self, v):
        self.auto_preview = bool(v)

    # ------------------------------------------------------------ editing
    def _push(self):
        if self.audio is None:
            return False
        if self.project_mode:
            self.info.emit("Multi-track mode: edit on the timeline (Multi-track tab). Open a file again to return to single-file editing.")
            return False
        self.undo.append(self.audio.copy())
        self.undo = self.undo[-8:]
        self.redo.clear()
        return True

    def _edited(self, keep_sel=False):
        if not keep_sel:
            self.sel = None
        self.mono_a, self.mono_b = self.audio.mean(axis=1).astype(np.float32), None
        self.preview, self.p_off = None, 0
        self.token += 1
        self.cache.clear()
        self.player.set_audio(self.audio, None)
        self.player.seek(0)
        self.set_ab("A")
        self._want_b = True
        self.audioChanged.emit()
        if self.auto_preview:
            self.debounce.start(300)

    def _range(self):
        if self.sel and self.sel[1] - self.sel[0] > 16:
            return int(self.sel[0]), int(self.sel[1])
        return None

    def trim(self):
        r = self._range()
        if r and self._push():
            self.audio = self.audio[r[0]:r[1]].copy()
            self._edited()

    def cut(self):
        r = self._range()
        if r and self._push():
            a, b = self.audio[:r[0]].copy(), self.audio[r[1]:].copy()
            if len(a) > 480:
                a[-240:] *= np.linspace(1, 0, 240)[:, None].astype(np.float32)
            if len(b) > 480:
                b[:240] *= np.linspace(0, 1, 240)[:, None].astype(np.float32)
            self.audio = np.vstack([a, b])
            self._edited()

    def silence(self):
        r = self._range()
        if r and self._push():
            n = min(240, (r[1] - r[0]) // 2)
            self.audio[r[0]:r[1]] = 0
            if r[0] > n:
                self.audio[r[0] - n:r[0]] *= np.linspace(1, 0, n)[:, None].astype(np.float32)
            self._edited(True)

    def fade(self, rising):
        if self.audio is None:
            return
        r = self._range() or (0, len(self.audio))
        if self._push():
            ramp = np.linspace(0, 1, r[1] - r[0], dtype=np.float32)
            self.audio[r[0]:r[1]] *= (ramp if rising else ramp[::-1])[:, None]
            self._edited(True)

    def gain(self, db):
        if self.audio is None:
            return
        r = self._range() or (0, len(self.audio))
        if self._push():
            self.audio[r[0]:r[1]] *= float(dsp.db2lin(db))
            self._edited(True)

    def normalize(self):
        if self.audio is None:
            return
        r = self._range() or (0, len(self.audio))
        pk = float(np.max(np.abs(self.audio[r[0]:r[1]])))
        if pk > 1e-6 and self._push():
            self.audio[r[0]:r[1]] *= float(dsp.db2lin(-1.0)) / pk
            self._edited(True)

    def reverse(self):
        if self.audio is None:
            return
        r = self._range() or (0, len(self.audio))
        if self._push():
            self.audio[r[0]:r[1]] = self.audio[r[0]:r[1]][::-1].copy()
            self._edited(True)

    def do_undo(self):
        if self.undo:
            self.redo.append(self.audio.copy())
            self.audio = self.undo.pop()
            self._edited()

    def do_redo(self):
        if self.redo:
            self.undo.append(self.audio.copy())
            self.audio = self.redo.pop()
            self._edited()

    # ------------------------------------------------------------ automatic cleanup state
    def apply_settings(self, settings_dict, label):
        """Apply a whole group of settings (Auto Clean or a preset) and remember where it came from."""
        self.cleaned_by, self.popup_done = label, False
        self.store.load(settings_dict)
        self.stateChanged.emit()

    def reset_to_original(self):
        if self.original is None:
            return
        keep = self.store.s
        s = cfg.preset_settings("Loudness Only")
        for k in ("out_channels", "out_format", "loudness_on", "loudness_target", "lufs_custom", "ceiling"):
            s[k] = keep[k]
        self.undo.clear()
        self.redo.clear()
        self.cleaned_by, self.popup_done = None, False
        self.store.load(s)
        if not self.project_mode:
            self.audio = self.original.copy()
            self._edited()
        else:
            self.preview = None
            self.cache.clear()
            self.token += 1
            if self.auto_preview:
                self.debounce.start(300)
        self.stateChanged.emit()

    # ------------------------------------------------------------ export
    def export_to(self, dst, settings_dict=None):
        if self.audio is None:
            self.error.emit("Open a file first.")
            return
        audio, s, src = self.audio.copy(), dict(settings_dict or self.store.s), self.src_path

        def work(prog, cancel):
            y, info = engine.render(audio, s, lambda f, m: prog(f * 0.9, m), cancel)
            prog(0.92, "Encoding")
            engine.export(y, dst, s, src, cancel)
            return info

        self.busy.emit("Exporting...")
        t = Task(work)
        self._hold(t)
        t.sig.progress.connect(self.progress.emit)
        t.sig.done.connect(lambda info: (self.idle.emit(), self.exported.emit(str(dst), info)))
        t.sig.failed.connect(lambda m: (self.idle.emit(), self.error.emit(f"Export failed:\n{m}")))
        self.pool.start(t)

    def shutdown(self):
        self.player.close()
