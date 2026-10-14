"""Skippable first-run tutorial: dimmed screen, an animated spotlight that glides between controls,
a pulsing highlight, a travelling pointer and a tip card."""
import math

from PySide6.QtCore import QEasingCurve, QPoint, QPointF, QRectF, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import theme


class TutorialOverlay(QWidget):
    finished = Signal(bool)   # True = completed, False = skipped

    def __init__(self, window):
        super().__init__(window)
        self.win = window
        self.steps, self.i = [], 0
        self.rect = QRectF(0, 0, 0, 0)
        self.r0 = self.r1 = QRectF()
        self.c0 = self.c1 = QPointF()
        self.t = 1.0
        self.pulse = 0.0
        self.title_text = ""
        self.anim = QVariantAnimation(self)
        self.anim.setDuration(650)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.anim.valueChanged.connect(self._step)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._pulse)
        self.setFocusPolicy(Qt.StrongFocus)

        self.card = QFrame(self)
        self.card.setObjectName("tutcard")
        v = QVBoxLayout(self.card)
        v.setContentsMargins(18, 14, 18, 14)
        self.h = QLabel()
        self.h.setWordWrap(True)
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.count = QLabel()
        row = QHBoxLayout()
        self.skip = QPushButton("Skip tutorial")
        self.back = QPushButton("Back")
        self.next = QPushButton("Next")
        self.next.setObjectName("primary")
        self.skip.clicked.connect(lambda: self.end(False))
        self.back.clicked.connect(lambda: self.goto(self.i - 1))
        self.next.clicked.connect(self._next)
        row.addWidget(self.skip)
        row.addStretch(1)
        row.addWidget(self.back)
        row.addWidget(self.next)
        v.addWidget(self.h)
        v.addWidget(self.body)
        v.addWidget(self.count)
        v.addLayout(row)
        self.card.setFixedWidth(380)
        self.hide()
        window.installEventFilter(self)

    # ------------------------------------------------------------------ control
    def start(self, steps, title="Tutorial"):
        self.steps, self.i, self.title_text = steps, 0, title
        self._style()
        self.setGeometry(self.win.rect())
        self.rect = QRectF(self.width() / 2, self.height() / 2, 0, 0)
        self.show()
        self.raise_()
        self.setFocus()
        self.timer.start(30)
        self.goto(0)

    def end(self, completed):
        self.timer.stop()
        self.anim.stop()
        self.hide()
        self.finished.emit(completed)

    def _next(self):
        if self.i >= len(self.steps) - 1:
            self.end(True)
        else:
            self.goto(self.i + 1)

    def eventFilter(self, obj, ev):
        if obj is self.win and ev.type() == ev.Type.Resize and self.isVisible():
            self.setGeometry(self.win.rect())
            self._aim(self.steps[self.i], animate=False)
        return False

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.end(False)
        elif e.key() in (Qt.Key_Right, Qt.Key_Return, Qt.Key_Space):
            self._next()
        elif e.key() == Qt.Key_Left:
            self.goto(self.i - 1)

    def mousePressEvent(self, e):
        e.accept()

    # ------------------------------------------------------------------ stepping
    def goto(self, i):
        if not (0 <= i < len(self.steps)):
            return
        self.i = i
        st = self.steps[i]
        if st.get("pre"):
            st["pre"]()
        self.h.setText(st["title"])
        self.body.setText(st["text"])
        self.count.setText(f"{i + 1} of {len(self.steps)}")
        self.back.setEnabled(i > 0)
        self.next.setText("Finish" if i == len(self.steps) - 1 else "Next")
        self.card.adjustSize()
        QTimer.singleShot(90, lambda: self._aim(st))

    def _target_rect(self, st):
        w = st["target"]() if st.get("target") else None
        if w is None or not w.isVisible():
            return QRectF(self.width() / 2 - 2, self.height() / 2 - 2, 4, 4)
        tl = w.mapTo(self.win, QPoint(0, 0))
        return QRectF(tl.x() - 8, tl.y() - 8, w.width() + 16, w.height() + 16)

    def _card_pos(self, r):
        self.card.adjustSize()
        cw, ch = self.card.width(), self.card.height()
        if r.width() <= 6:                       # no target: centre the card
            return QPointF((self.width() - cw) / 2, (self.height() - ch) / 2)
        x = min(max(12, r.center().x() - cw / 2), self.width() - cw - 12)
        if r.bottom() + ch + 18 < self.height():
            y = r.bottom() + 16
        elif r.top() - ch - 18 > 0:
            y = r.top() - ch - 16
        else:
            y = max(12, min(self.height() - ch - 12, r.center().y() - ch / 2))
            x = r.right() + 16 if r.right() + cw + 16 < self.width() else max(12, r.left() - cw - 16)
        return QPointF(x, y)

    def _aim(self, st, animate=True):
        self.r0 = QRectF(self.rect)
        self.c0 = QPointF(self.card.pos())
        self.r1 = self._target_rect(st)
        self.c1 = self._card_pos(self.r1)
        self.anim.stop()
        if animate:
            self.t = 0.0
            self.anim.start()
        else:
            self._step(1.0)

    def _step(self, t):
        t = float(t)
        self.t = t
        lerp = lambda a, b: a + (b - a) * t
        self.rect = QRectF(lerp(self.r0.x(), self.r1.x()), lerp(self.r0.y(), self.r1.y()),
                           lerp(self.r0.width(), self.r1.width()), lerp(self.r0.height(), self.r1.height()))
        self.card.move(int(lerp(self.c0.x(), self.c1.x())), int(lerp(self.c0.y(), self.c1.y())))
        self.update()

    def _pulse(self):
        self.pulse = (self.pulse + 0.04) % 1.0
        self.update()

    def _style(self):
        p = theme.palette()
        self.card.setStyleSheet(
            f"QFrame#tutcard {{ background: {p['panel']}; border: 2px solid {p['accent']}; border-radius: 14px; }}"
            f"QFrame#tutcard QLabel {{ background: transparent; color: {p['text']}; font-size: 13px; }}")
        self.h.setStyleSheet(f"font-size: 17px; font-weight: 700; color: {p['accent']}; background: transparent;")
        self.count.setStyleSheet(f"color: {p['muted']}; font-size: 11px; background: transparent;")

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRect(QRectF(self.rect.__class__(0, 0, self.width(), self.height())))
        if self.rect.width() > 6:
            hole = QPainterPath()
            hole.addRoundedRect(self.rect, 12, 12)
            path = path.subtracted(hole)
        p.fillPath(path, QColor(6, 10, 16, 175))
        if self.rect.width() > 6:
            glow = 3 + 3 * math.sin(self.pulse * 2 * math.pi)
            c = theme.color("accent")
            for k, a in ((glow + 6, 40), (glow + 2, 90)):
                p.setPen(QPen(QColor(c.red(), c.green(), c.blue(), a), k))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(self.rect, 12, 12)
            p.setPen(QPen(c, 2.5))
            p.drawRoundedRect(self.rect, 12, 12)
            # travelling pointer with a tap ripple
            tgt = self.rect.center()
            start = QPointF(self.width() * 0.5, self.height() * 0.5)
            pt = QPointF(start.x() + (tgt.x() - start.x()) * min(1.0, self.t * 1.15), start.y() + (tgt.y() - start.y()) * min(1.0, self.t * 1.15))
            if self.t >= 1.0:
                r = 8 + 26 * self.pulse
                p.setPen(QPen(QColor(c.red(), c.green(), c.blue(), int(200 * (1 - self.pulse))), 2.5))
                p.drawEllipse(tgt, r, r)
                pt = tgt
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 230))
            p.drawEllipse(pt, 9, 9)
