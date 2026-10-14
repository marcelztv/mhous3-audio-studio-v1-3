"""Generates the logo files (dark grey, light blue and white)."""
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "assets"
S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
# dark-grey rounded tile with a vertical gradient
tile = Image.new("RGBA", (S, S))
td = ImageDraw.Draw(tile)
for y in range(S):
    t = y / S
    c = (int(40 + 14 * t), int(44 + 14 * t), int(51 + 16 * t), 255)
    td.line([(0, y), (S, y)], fill=c)
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle((16, 16, S - 16, S - 16), radius=210, fill=255)
img.paste(tile, (0, 0), mask)
d.rounded_rectangle((16, 16, S - 16, S - 16), radius=210, outline=(92, 182, 255, 255), width=18)
bars = [0.16, 0.34, 0.58, 0.86, 0.52, 0.95, 0.66, 0.38, 0.6, 0.28, 0.14]
w, gap = 52, 28
x = (S - (len(bars) * w + (len(bars) - 1) * gap)) // 2
for i, b in enumerate(bars):
    h = int(620 * b)
    y0 = S // 2 - h // 2
    col = (92, 182, 255, 255) if i % 2 == 0 else (255, 255, 255, 255)
    d.rounded_rectangle((x, y0, x + w, y0 + h), radius=26, fill=col)
    x += w + gap
img.resize((512, 512), Image.LANCZOS).save(OUT / "icon.png")
img.save(OUT / "icon.ico", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
print("assets ok")
