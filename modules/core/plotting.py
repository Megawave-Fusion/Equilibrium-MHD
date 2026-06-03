"""Small plotting helpers for simulation diagnostics.

The project intentionally keeps GUI/runtime dependencies light, so these
helpers use Pillow instead of matplotlib.  They produce labelled field maps and
profile/projection plots that mirror common integrated-modelling outputs:
2-D state maps plus mid-plane, axial, or radial profile curves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None
    ImageDraw = None
    ImageFont = None

Array = np.ndarray


COLORS = {
    "bg": (248, 250, 252),
    "panel": (255, 255, 255),
    "ink": (18, 28, 35),
    "muted": (73, 88, 98),
    "grid": (198, 210, 220),
    "blue": (28, 90, 158),
    "red": (196, 67, 61),
    "green": (26, 127, 85),
    "gold": (150, 111, 35),
    "purple": (104, 83, 169),
}


def _font(size: int = 13, bold: bool = False):
    if ImageFont is None:
        return None
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _finite_range(values: Array, signed: bool = False) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    if signed:
        limit = float(np.max(np.abs(finite)))
        return -limit, limit if limit > 0 else 1.0
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _color(norm: float, signed: bool = False) -> tuple[int, int, int]:
    norm = float(np.clip(norm, 0.0, 1.0))
    if signed:
        if norm < 0.5:
            t = norm / 0.5
            return (int(55 + 150 * t), int(92 + 120 * t), int(170 + 55 * t))
        t = (norm - 0.5) / 0.5
        return (int(225 - 25 * (1 - t)), int(218 - 150 * t), int(212 - 150 * t))
    return (
        int(35 + 205 * norm),
        int(70 + 125 * (1.0 - abs(norm - 0.55))),
        int(120 + 105 * (1.0 - norm)),
    )


def _heatmap(values: Array, width: int, height: int, signed: bool = False) -> "Image.Image":
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D field, got shape {arr.shape}")
    lo, hi = _finite_range(arr, signed=signed)
    norm = np.clip((arr - lo) / max(hi - lo, 1.0e-30), 0.0, 1.0)
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    for channel_values in np.ndindex(arr.shape):
        rgb[channel_values] = _color(float(norm[channel_values]), signed=signed)
    image = Image.fromarray(np.flipud(np.swapaxes(rgb, 0, 1)), "RGB")
    return image.resize((width, height))


def _draw_axes(draw, box: tuple[int, int, int, int], x_label: str, y_label: str, title: str) -> None:
    title_font = _font(14, bold=True)
    small_font = _font(11)
    left, top, right, bottom = box
    draw.rectangle(box, outline=COLORS["grid"], width=1)
    draw.text((left, top - 22), title, fill=COLORS["ink"], font=title_font)
    draw.text(((left + right) // 2 - 28, bottom + 8), x_label, fill=COLORS["muted"], font=small_font)
    draw.text((left - 30, (top + bottom) // 2 - 8), y_label, fill=COLORS["muted"], font=small_font)


def _draw_colorbar(draw, box: tuple[int, int, int, int], lo: float, hi: float, signed: bool, unit: str) -> None:
    left, top, right, bottom = box
    for y in range(top, bottom):
        norm = 1.0 - (y - top) / max(bottom - top - 1, 1)
        draw.line((left, y, right, y), fill=_color(norm, signed=signed))
    small_font = _font(10)
    draw.rectangle(box, outline=COLORS["grid"])
    draw.text((right + 4, top - 2), f"{hi:.3g}", fill=COLORS["muted"], font=small_font)
    draw.text((right + 4, bottom - 10), f"{lo:.3g}", fill=COLORS["muted"], font=small_font)
    if unit:
        draw.text((left - 4, bottom + 6), unit, fill=COLORS["muted"], font=small_font)


def _draw_line_panel(
    draw,
    box: tuple[int, int, int, int],
    x: Array,
    series: Iterable[tuple[str, Array, tuple[int, int, int]]],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, fill=COLORS["panel"], outline=COLORS["grid"])
    title_font = _font(13, bold=True)
    small_font = _font(10)
    draw.text((left, top - 20), title, fill=COLORS["ink"], font=title_font)
    draw.text(((left + right) // 2 - 30, bottom + 8), x_label, fill=COLORS["muted"], font=small_font)
    draw.text((left - 28, (top + bottom) // 2 - 8), y_label, fill=COLORS["muted"], font=small_font)
    x_arr = np.asarray(x, dtype=float)
    valid_series = []
    y_all = []
    for name, values, color in series:
        y_arr = np.asarray(values, dtype=float)
        if y_arr.size != x_arr.size:
            continue
        finite = y_arr[np.isfinite(y_arr)]
        if finite.size:
            valid_series.append((name, y_arr, color))
            y_all.append(finite)
    if not valid_series:
        draw.text((left + 18, top + 46), "no finite profile", fill=COLORS["muted"], font=small_font)
        return
    x_lo, x_hi = _finite_range(x_arr)
    y_lo, y_hi = _finite_range(np.concatenate(y_all))
    for tick in np.linspace(0.0, 1.0, 5):
        px = left + tick * (right - left)
        py = top + tick * (bottom - top)
        draw.line((px, top, px, bottom), fill=(232, 238, 244))
        draw.line((left, py, right, py), fill=(232, 238, 244))
    for idx, (name, y_arr, color) in enumerate(valid_series):
        pts = []
        for xv, yv in zip(x_arr, y_arr):
            if not np.isfinite(xv) or not np.isfinite(yv):
                continue
            px = left + (float(xv) - x_lo) / max(x_hi - x_lo, 1.0e-30) * (right - left)
            py = bottom - (float(yv) - y_lo) / max(y_hi - y_lo, 1.0e-30) * (bottom - top)
            pts.append((px, py))
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=3)
        draw.text((left + 12 + 130 * (idx % 4), bottom - 18 - 16 * (idx // 4)), name, fill=color, font=small_font)
    draw.text((left + 8, top + 6), f"y={y_lo:.3g}..{y_hi:.3g}", fill=COLORS["muted"], font=small_font)


def _draw_single_line_panel(
    draw,
    box: tuple[int, int, int, int],
    x: Array,
    y: Array,
    title: str,
    x_label: str,
    y_label: str,
    color: tuple[int, int, int],
) -> None:
    _draw_line_panel(draw, box, x, [(title, y, color)], title, x_label, y_label)


def _normalized(values: Array) -> Array:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def save_2d_projection_figure(
    path: Path,
    x: Array,
    y: Array,
    fields: list[dict[str, object]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    mid_y_label: str,
    vertical_x_label: str,
) -> None:
    """Write 2-D fields with labelled axes plus two projection/profile panels."""
    if Image is None or ImageDraw is None:
        return
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y axes must be 1-D")
    usable = [field for field in fields if np.asarray(field["values"]).shape == (x_arr.size, y_arr.size)]
    if not usable:
        return
    usable = usable[:4]
    projection_rows = len(usable)
    canvas_height = 360 + projection_rows * 250
    canvas = Image.new("RGB", (1280, canvas_height), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 28), title, fill=COLORS["ink"], font=_font(22, bold=True))
    draw.text(
        (36, 58),
        "Top: labelled 2-D field maps. Bottom: one variable per projection/profile panel.",
        fill=COLORS["muted"],
        font=_font(12),
    )
    panel_w, panel_h = 255, 185
    start_x, start_y, gap = 52, 112, 54
    for idx, field in enumerate(usable):
        left = start_x + idx * (panel_w + gap)
        top = start_y
        values = np.asarray(field["values"], dtype=float)
        signed = bool(field.get("signed", False))
        img = _heatmap(values, panel_w, panel_h, signed=signed)
        canvas.paste(img, (left, top))
        _draw_axes(draw, (left, top, left + panel_w, top + panel_h), x_label, y_label, str(field["name"]))
        lo, hi = _finite_range(values, signed=signed)
        _draw_colorbar(draw, (left + panel_w + 8, top, left + panel_w + 18, top + panel_h), lo, hi, signed, str(field.get("unit", "")))
    mid_index = int(np.argmin(np.abs(y_arr)))
    axis_index = int(np.argmin(np.abs(x_arr - np.median(x_arr))))
    palette = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["gold"]]
    rho = np.linspace(0.0, 1.0, x_arr.size)
    for idx, field in enumerate(usable):
        values = np.asarray(field["values"], dtype=float)
        scale = float(field.get("scale", 1.0))
        color = palette[idx % len(palette)]
        top = 392 + idx * 250
        projection_boxes = (
            (68, top, 360, top + 148),
            (484, top, 776, top + 148),
            (900, top, 1192, top + 148),
        )
        for box, x_axis, profile, label, axis_label in (
            (
                projection_boxes[0],
                x_arr,
                _normalized(values[:, mid_index] * scale),
                f"{field['name']} | {mid_y_label}",
                x_label,
            ),
            (
                projection_boxes[1],
                y_arr,
                _normalized(values[axis_index, :] * scale),
                f"{field['name']} | {vertical_x_label}",
                y_label,
            ),
            (
                projection_boxes[2],
                rho,
                _normalized(np.nanmean(values, axis=1) * scale),
                f"{field['name']} | radial mean",
                "rho_N",
            ),
        ):
            _draw_single_line_panel(
                draw,
                box,
                x_axis,
                profile,
                str(label),
                axis_label,
                "normalized value",
                color,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_profile_figure(
    path: Path,
    x: Array,
    series: list[tuple[str, Array]],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Write labelled profile panels, one physical quantity per panel."""
    if Image is None or ImageDraw is None:
        return
    panel_count = max(1, len(series))
    columns = 2 if panel_count > 1 else 1
    rows = int(np.ceil(panel_count / columns))
    width = 1180
    height = 130 + rows * 240
    canvas = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 30), title, fill=COLORS["ink"], font=_font(22, bold=True))
    draw.text((42, 58), "Each panel shows one output quantity to keep units and scale readable.", fill=COLORS["muted"], font=_font(12))
    palette = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["gold"], COLORS["purple"]]
    box_w = 492 if columns == 2 else 1024
    box_h = 164
    for idx, (name, values) in enumerate(series):
        col = idx % columns
        row = idx // columns
        left = 74 + col * 560
        top = 126 + row * 240
        _draw_single_line_panel(
            draw,
            (left, top, left + box_w, top + box_h),
            np.asarray(x, dtype=float),
            np.asarray(values, dtype=float),
            str(name),
            x_label,
            y_label,
            palette[idx % len(palette)],
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
