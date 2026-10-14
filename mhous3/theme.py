"""Dark / Light / System themes.  Palette: dark grey, light blue, white."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication

PALETTES = {
    "dark": dict(bg="#1c1e22", panel="#24272c", panel2="#2c3036", border="#3a3f47", text="#eef2f7", muted="#9aa5b4",
                 accent="#5cb6ff", accent_hi="#8dcbff", on_accent="#08141f", wave="#6f7a8a", grid="#30353c",
                 good="#4cd4a0", warn="#ffb454", danger="#ff6b6b", card="#2a2e34", card_sel="#243a50", scrim="#0b0c0e"),
    "light": dict(bg="#e9edf2", panel="#ffffff", panel2="#f2f5f9", border="#cfd6e0", text="#1b2430", muted="#637083",
                  accent="#1f84e0", accent_hi="#4aa0ee", on_accent="#ffffff", wave="#8996a7", grid="#dde3ea",
                  good="#1fa57a", warn="#d98a1a", danger="#d94b4b", card="#f7f9fc", card_sel="#dcebfb", scrim="#2a3342"),
}
_state = {"name": "dark", "mode": "dark"}


def palette():
    return PALETTES[_state["name"]]


def color(key, alpha=None):
    c = QColor(palette()[key])
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def current_name():
    return _state["name"]


def resolve(mode):
    if mode == "system":
        try:
            cs = QGuiApplication.styleHints().colorScheme()
            return "light" if cs == Qt.ColorScheme.Light else "dark"
        except Exception:
            return "dark"
    return mode if mode in PALETTES else "dark"


TEMPLATE = """
* { font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif; font-size: 13px; }
QMainWindow, QDialog, QWidget { background: @bg@; color: @text@; }
QMainWindow::separator { background: @border@; width: 2px; height: 2px; }
QToolBar#ribbon { background: @panel@; border: none; border-bottom: 1px solid @border@; spacing: 4px; padding: 5px 8px; }
QToolBar#tools { background: @panel@; border: none; border-right: 1px solid @border@; spacing: 3px; padding: 6px 4px; }
QToolBar QToolButton, QToolButton { background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 6px 10px; color: @text@; }
QToolBar QToolButton:hover, QToolButton:hover { background: @panel2@; border-color: @border@; }
QToolButton:checked, QToolButton#wsbtn:checked { background: @accent@; color: @on_accent@; font-weight: 700; border-color: @accent@; }
QToolButton#wsbtn { padding: 7px 16px; border-radius: 7px; }
QToolButton::menu-indicator { image: none; }
QDockWidget { titlebar-close-icon: none; color: @muted@; font-weight: 600; }
QDockWidget::title { background: @panel2@; padding: 7px 10px; border-bottom: 1px solid @border@; text-align: left; }
QTabWidget::pane { border: 1px solid @border@; top: -1px; background: @panel@; }
QTabBar::tab { background: @panel2@; padding: 8px 18px; border: 1px solid @border@; border-bottom: none; margin-right: 2px; color: @muted@; }
QTabBar::tab:selected { background: @panel@; color: @accent@; font-weight: 600; }
QGroupBox { background: @panel@; border: 1px solid @border@; border-radius: 8px; margin-top: 14px; padding: 10px 8px 6px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: @accent@; font-weight: 600; }
QPushButton { background: @panel2@; border: 1px solid @border@; border-radius: 6px; padding: 6px 14px; color: @text@; }
QPushButton:hover { border-color: @accent@; background: @card@; }
QPushButton:disabled { color: @muted@; }
QPushButton:checked { background: @accent@; color: @on_accent@; font-weight: 700; border-color: @accent@; }
QPushButton#primary { background: @accent@; color: @on_accent@; font-weight: 700; border: none; padding: 9px 20px; }
QPushButton#primary:hover { background: @accent_hi@; }
QPushButton#hero { background: @accent@; color: @on_accent@; font-weight: 700; font-size: 15px; border: none; border-radius: 10px; padding: 12px 26px; }
QPushButton#hero:hover { background: @accent_hi@; }
QFrame#card { background: @card@; border: 1px solid @border@; border-radius: 10px; }
QFrame#card:hover { border-color: @accent@; }
QFrame#card[selected="true"] { background: @card_sel@; border: 2px solid @accent@; }
QFrame#card QLabel { background: transparent; }
QLabel#cardtitle { font-weight: 700; font-size: 13px; }
QLabel#carddesc { color: @muted@; font-size: 11px; }
QLabel#hint { color: @muted@; }
QLabel#info { background: @panel@; border: 1px solid @border@; border-radius: 6px; padding: 7px 10px; }
QComboBox, QTableWidget, QTextBrowser, QKeySequenceEdit, QLineEdit, QSpinBox { background: @panel@; border: 1px solid @border@; border-radius: 5px; padding: 4px 8px; color: @text@; }
QComboBox QAbstractItemView { background: @panel@; selection-background-color: @accent@; selection-color: @on_accent@; border: 1px solid @border@; }
QSlider::groove:horizontal { height: 4px; background: @border@; border-radius: 2px; }
QSlider::sub-page:horizontal { background: @accent@; border-radius: 2px; }
QSlider::handle:horizontal { background: @text@; width: 14px; margin: -6px 0; border-radius: 7px; border: 2px solid @accent@; }
QProgressBar { background: @panel@; border: 1px solid @border@; border-radius: 5px; text-align: center; height: 16px; color: @text@; }
QProgressBar::chunk { background: @accent@; border-radius: 4px; }
QHeaderView::section { background: @panel2@; border: none; border-bottom: 1px solid @border@; padding: 6px; color: @muted@; }
QTableWidget { gridline-color: @border@; alternate-background-color: @panel2@; selection-background-color: @card_sel@; selection-color: @text@; }
QScrollBar:horizontal { height: 12px; background: transparent; }
QScrollBar::handle:horizontal { background: @border@; border-radius: 5px; min-width: 30px; }
QScrollBar:vertical { width: 12px; background: transparent; }
QScrollBar::handle:vertical { background: @border@; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollArea { border: none; background: transparent; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid @muted@; border-radius: 4px; background: @panel@; }
QCheckBox::indicator:checked { background: @accent@; border-color: @accent@; }
QToolTip { background: @panel2@; color: @text@; border: 1px solid @accent@; padding: 6px 8px; }
QStatusBar { background: @panel@; color: @muted@; border-top: 1px solid @border@; }
QMenuBar { background: @panel@; } QMenuBar::item:selected { background: @panel2@; }
QMenu { background: @panel@; border: 1px solid @border@; } QMenu::item { padding: 6px 24px; } QMenu::item:selected { background: @accent@; color: @on_accent@; }
QMenu::separator { height: 1px; background: @border@; margin: 4px 8px; }
"""


def stylesheet(p):
    css = TEMPLATE
    for k, v in p.items():
        css = css.replace(f"@{k}@", v)
    return css


def apply(app, mode):
    _state["mode"] = mode
    _state["name"] = resolve(mode)
    app.setStyleSheet(stylesheet(palette()))
    return _state["name"]
