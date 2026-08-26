"""Картинка социального превью: 1280×640, как требует GitHub.

**Шрифт подменён и это надо знать.** Гарнитуры проекта лежат в woff2,
а PIL их не читает; конвертера в системе нет. Набрано Georgia — тот же
класс серифа, что Source Serif 4, и на кегле превью разница не бросается.
Когда появится конвертер, набор повторяется настоящей гарнитурой.

Марка рисуется теми же модулями, что docs/assets/logo.svg: одна раскладка,
два вывода.
"""
# Раскладка марки — одна на все выводы: docs/assets/logo.svg, вордмарк,
# компонент интерфейса и эта картинка рисуют одни и те же девять модулей.
PITCH, MODULE = 4.8, 3.84          # то же отношение, что у знака ragworld
ROWS = (5, 3, 1)                   # найдено, пережило реранкер, дошло до ответа


def modules(box=24.0):
    w = (max(ROWS) - 1) * PITCH + MODULE
    h = (len(ROWS) - 1) * PITCH + MODULE
    ox, oy = (box - w) / 2, (box - h) / 2
    out = []
    for r, n in enumerate(ROWS):
        start = ox + (max(ROWS) - n) / 2 * PITCH
        for c in range(n):
            out.append((round(start + c * PITCH, 2), round(oy + r * PITCH, 2)))
    return out

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
BG, INK, MUTED = (14, 16, 19), (232, 234, 237), (155, 161, 172)
G = "/System/Library/Fonts/Supplemental/Georgia"

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# Марка: те же девять модулей, масштаб от 24 к 190.
scale = 190 / 24
# Композиция выравнивается по центру: превью обрезают по-разному, и блок,
# прижатый к левому краю, теряет половину при обрезке справа.
CONTENT_W = 190 + 64 + 640          # марка, зазор, самая длинная строка
mx = (W - CONTENT_W) // 2
my = 168
for x, y in modules():
    d.rectangle([mx + x*scale, my + y*scale,
                 mx + (x + MODULE)*scale - 1, my + (y + MODULE)*scale - 1], fill=INK)

title = ImageFont.truetype(f"{G} Bold.ttf", 82)
lead  = ImageFont.truetype(f"{G}.ttf", 38)
small = ImageFont.truetype(f"{G}.ttf", 26)

tx = mx + 24*scale + 64
d.text((tx, 214), "Causa RAG", font=title, fill=INK)
# Строка из README: она же и обещание платформы.
d.text((tx, 318), "Shows which questions a change fixed,", font=lead, fill=MUTED)
d.text((tx, 366), "and which it broke.", font=lead, fill=MUTED)
d.text((tx, 442), "A diagnostic bench for retrieval-augmented generation.",
       font=small, fill=MUTED)
d.text((tx, 476), "Runs entirely on your own machine.", font=small, fill=MUTED)

# Полоска акцентов Okabe-Ito понизу: та же палитра, что у диаграмм платформы.
# Полоска встаёт под текстом, а не в углу: одинокий элемент у края читается
# обрезком чего-то, а не частью композиции.
for i, c in enumerate(("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00")):
    d.rectangle([tx + i*40, 534, tx + i*40 + 28, 542],
                fill=tuple(int(c[j:j+2], 16) for j in (1, 3, 5)))

out = Path(__file__).resolve().parent.parent / "docs/assets/social-preview.png"
img.save(out)
print(out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out, img.size)
