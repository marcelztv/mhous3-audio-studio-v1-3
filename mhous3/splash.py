"""Start-up splash: quarter of the screen (half width x half height), centred, logo + 'Built for M Hous3 Media'
and an animated, typing 'running text' of what is loading."""
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QLinearGradient, QPainter, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import QWidget

from . import APP_NAME, APP_VERSION
from .paths import app_dir


class Splash(QWidget):
    def __init__(self):
        super().__init__(None, Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        scr = QGuiApplication.primaryScreen().availableGeometry()
        w, h = max(640, scr.width() // 2), max(380, scr.height() // 2)      # half x half = a quarter of the screen
        self.setGeometry(scr.x() + (scr.width() - w) // 2, scr.y() + (scr.height() - h) // 2, w, h)
        logo = app_dir() / "assets" / "icon.png"
        self.logo = QPixmap(str(logo)) if logo.exists() else QPixmap()
        self.lines, self.typed, self.phase, self.frac = [], 0, 0.0, 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(28)

    def message(self, text, frac=None):
        self.lines.append(text)
        self.lines = self.lines[-4:]
        self.typed = 0
        if frac is not None:
            self.frac = frac
        self.update()

    def _tick(self):
        self.phase = (self.phase + 0.018) % 1.0
        if self.lines and self.typed < len(self.lines[-1]) + 1:
            self.typed += 1
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        W, H = self.width(), self.height()
        body = QRectF(6, 6, W - 12, H - 12)
        g = QLinearGradient(0, 0, W, H)
        g.setColorAt(0, QColor("#1b1d21"))
        g.setColorAt(1, QColor("#2a2e35"))
        p.setPen(QPen(QColor("#3f4650"), 1.5))
        p.setBrush(g)
        p.drawRoundedRect(body, 22, 22)
        rg = QRadialGradient(W / 2, H * 0.30, H * 0.5)
        rg.setColorAt(0, QColor(92, 182, 255, 70))
        rg.setColorAt(1, QColor(92, 182, 255, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(rg)
        p.drawEllipse(QRectF(W / 2 - H * 0.5, H * 0.30 - H * 0.5, H, H))
        lh = H * 0.30
        if not self.logo.isNull():
            p.drawPixmap(QRectF(W / 2 - lh / 2, H * 0.08, lh, lh), self.logo, QRectF(self.logo.rect()))
        f = QFont("Segoe UI")
        f.setPixelSize(int(H * 0.075))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#ffffff"))
        p.drawText(QRectF(0, H * 0.40, W, H * 0.10), Qt.AlignCenter, APP_NAME)
        f.setPixelSize(int(H * 0.040))
        f.setBold(False)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 3)
        p.setFont(f)
        p.setPen(QColor("#5cb6ff"))
        p.drawText(QRectF(0, H * 0.51, W, H * 0.06), Qt.AlignCenter, "BUILT FOR M HOUS3 MEDIA")
        # running text (typewriter, older lines fade upwards)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 0)
        f.setPixelSize(int(H * 0.034))
        f.setFamily("Consolas")
        p.setFont(f)
        y0 = H * 0.66
        for k, line in enumerate(self.lines):
            last = k == len(self.lines) - 1
            age = len(self.lines) - 1 - k
            txt = line[: self.typed] + ("_" if int(self.phase * 8) % 2 == 0 else " ") if last else line
            c = QColor("#dfe7f1" if last else "#8fa0b4")
            c.setAlpha(255 if last else max(40, 170 - age * 55))
            p.setPen(c)
            p.drawText(QRectF(W * 0.12, y0 + (k - len(self.lines) + 1) * H * 0.045 + H * 0.045, W * 0.76, H * 0.05),
                       Qt.AlignLeft | Qt.AlignVCenter, ("> " if last else "  ") + txt)
        # progress with a shimmer
        bx, bw, by = W * 0.12, W * 0.76, H * 0.90
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#3a3f47"))
        p.drawRoundedRect(QRectF(bx, by, bw, 5), 2.5, 2.5)
        p.setBrush(QColor("#5cb6ff"))
        p.drawRoundedRect(QRectF(bx, by, bw * max(0.04, self.frac), 5), 2.5, 2.5)
        sx = bx + (bw * max(0.04, self.frac) + 80) * self.phase - 80
        sg = QLinearGradient(sx, 0, sx + 80, 0)
        sg.setColorAt(0, QColor(255, 255, 255, 0))
        sg.setColorAt(0.5, QColor(255, 255, 255, 150))
        sg.setColorAt(1, QColor(255, 255, 255, 0))
        p.setBrush(sg)
        p.drawRoundedRect(QRectF(max(bx, sx), by, min(80, bx + bw * max(0.04, self.frac) - max(bx, sx)), 5), 2.5, 2.5) if sx < bx + bw * max(0.04, self.frac) else None
        f.setFamily("Segoe UI")
        f.setPixelSize(int(H * 0.028))
        p.setFont(f)
        p.setPen(QColor("#6f7b8b"))
        p.drawText(QRectF(0, H * 0.935, W, H * 0.05), Qt.AlignCenter, f"Version {APP_VERSION}   |   (c) M Hous3 Media. All rights reserved.")
