"""Start-up: show the splash straight away, load the heavy libraries on a worker thread (the splash keeps
animating), then build the main window."""
import sys
import time


def run():
    sys.setswitchinterval(0.0005)         # lets the interface thread win the interpreter lock quickly = no stalls
    from PySide6.QtCore import QThread, QTimer, Signal
    from PySide6.QtWidgets import QApplication, QMessageBox

    from . import APP_NAME, theme
    from .config import Config
    from .splash import Splash

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("M Hous3 Media")
    cfg = Config()
    theme.apply(app, cfg.get("theme", "system"))
    t0 = time.monotonic()
    splash = Splash()
    splash.show()
    app.processEvents()

    class Loader(QThread):
        step = Signal(str, float)
        failed = Signal(str)

        def run(self):
            try:
                self.step.emit("Waking the audio engine...", 0.10)
                import numpy  # noqa: F401
                self.step.emit("Loading signal processing...", 0.30)
                import scipy.signal, scipy.ndimage, scipy.interpolate  # noqa: F401,E401
                self.step.emit("Loading sound file support...", 0.50)
                import soundfile  # noqa: F401
                try:
                    import sounddevice  # noqa: F401
                except Exception:
                    pass
                self.step.emit("Loading the cinema DSP library...", 0.65)
                import mhous3.dsp, mhous3.engine  # noqa: F401,E401
                self.step.emit("Loading the Auto Clean analyser...", 0.78)
                import mhous3.analyzer  # noqa: F401
                self.step.emit("Preparing the workspaces...", 0.90)
                import mhous3.mainwindow  # noqa: F401
            except Exception as e:                      # pragma: no cover
                self.failed.emit(f"{type(e).__name__}: {e}")

    loader = Loader()
    holder, failed = {}, []
    loader.step.connect(lambda m, f: splash.message(m, f))
    loader.failed.connect(lambda m: (failed.append(m), splash.close(), QMessageBox.critical(None, APP_NAME, f"Could not start:\n{m}"), app.quit()))

    def build():
        from .mainwindow import MainWindow
        win = MainWindow(cfg, progress=lambda m: (splash.message(m, 0.95), app.processEvents()))
        holder["win"] = win
        splash.message("Ready.", 1.0)
        delay = max(250, int((1.7 - (time.monotonic() - t0)) * 1000))
        QTimer.singleShot(delay, lambda: (win.show(), splash.close(), win.after_show()))

    loader.finished.connect(lambda: None if failed else build())
    loader.start()
    code = app.exec()
    sys.exit(code)
