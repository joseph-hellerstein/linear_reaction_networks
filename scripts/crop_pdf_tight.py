"""Crop PDF pages to tighter bounds.

Two methods are supported:
1. `boxes`: uses existing page boxes (`cropbox`, `trimbox`, `bleedbox`, `artbox`,
    `mediabox`) and intersects them.
2. `render`: renders each page and computes a non-white pixel bounding box for
    true content-aware cropping.

`auto` tries `render` first, then falls back to `boxes`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from pypdf import PdfReader, PdfWriter  # type: ignore
from pypdf.generic import RectangleObject  # type: ignore


def _rectangle_values(box: RectangleObject) -> tuple[float, float, float, float]:
    return (
        float(box.left),
        float(box.bottom),
        float(box.right),
        float(box.top),
    )


def _is_valid_rectangle(rect: tuple[float, float, float, float]) -> bool:
    left, bottom, right, top = rect
    return right > left and top > bottom


def tightest_box(page) -> Optional[tuple[float, float, float, float]]:
    candidates: List[tuple[float, float, float, float]] = []
    for name in ["cropbox", "trimbox", "bleedbox", "artbox", "mediabox"]:
        box = getattr(page, name, None)
        if box is None:
            continue
        rect = _rectangle_values(box)
        if _is_valid_rectangle(rect):
            candidates.append(rect)

    if not candidates:
        return None

    left = max(rect[0] for rect in candidates)
    bottom = max(rect[1] for rect in candidates)
    right = min(rect[2] for rect in candidates)
    top = min(rect[3] for rect in candidates)

    rect = (left, bottom, right, top)
    if not _is_valid_rectangle(rect):
        return None
    return rect


def rendered_content_box(
    fitz_page,
    white_threshold: int = 250,
) -> Optional[tuple[float, float, float, float]]:
    import numpy as np

    pix = fitz_page.get_pixmap(alpha=False)
    data = np.frombuffer(pix.samples, dtype=np.uint8)
    data = data.reshape(pix.height, pix.width, pix.n)
    rgb = data[:, :, :3]
    non_white = (rgb < white_threshold).any(axis=2)

    ys, xs = np.where(non_white)
    if xs.size == 0 or ys.size == 0:
        return None

    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1

    page_rect = fitz_page.rect
    x_scale = float(page_rect.width) / float(pix.width)
    y_scale = float(page_rect.height) / float(pix.height)

    left = float(page_rect.x0) + x_min * x_scale
    right = float(page_rect.x0) + x_max * x_scale
    top = float(page_rect.y1) - y_min * y_scale
    bottom = float(page_rect.y1) - y_max * y_scale

    rect = (left, bottom, right, top)
    if not _is_valid_rectangle(rect):
        return None
    return rect


def apply_crop(page, rect: tuple[float, float, float, float], padding: float = 0.0) -> None:
    left, bottom, right, top = rect
    padded = RectangleObject((left - padding, bottom - padding, right + padding, top + padding))
    page.cropbox = padded
    page.trimbox = padded
    page.bleedbox = padded
    page.artbox = padded


def crop_pdf(
    input_pdf: Path,
    output_pdf: Path,
    padding: float = 0.0,
    pages: Optional[Iterable[int]] = None,
    method: str = "auto",
    white_threshold: int = 250,
) -> None:
    reader = PdfReader(str(input_pdf))
    writer = PdfWriter()

    selected_pages = set(pages) if pages is not None else None

    fitz_doc = None
    if method in {"render", "auto"}:
        try:
            import fitz  # type: ignore

            fitz_doc = fitz.open(str(input_pdf))
        except ImportError:
            if method == "render":
                raise RuntimeError("Render mode requires PyMuPDF (`pip install pymupdf`).")

    for index, page in enumerate(reader.pages):
        if selected_pages is None or index in selected_pages:
            rect = None
            if method in {"render", "auto"} and fitz_doc is not None:
                rect = rendered_content_box(fitz_doc[index], white_threshold=white_threshold)
            if rect is None and method in {"boxes", "auto"}:
                rect = tightest_box(page)
            if rect is not None:
                apply_crop(page, rect, padding=padding)
        writer.add_page(page)

    if fitz_doc is not None:
        fitz_doc.close()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as stream:
        writer.write(stream)


def parse_page_list(raw: str) -> List[int]:
    result: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid page range: {token}")
            result.extend(range(start - 1, end))
        else:
            result.append(int(token) - 1)
    unique_sorted = sorted(set(result))
    if any(page < 0 for page in unique_sorted):
        raise ValueError("Pages are 1-based and must be positive integers.")
    return unique_sorted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop pages to the tightest available PDF page box using pypdf.",
    )
    parser.add_argument("input", type=Path, help="Path to input PDF")
    parser.add_argument("output", type=Path, help="Path to output PDF")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Padding (in PDF points) added around computed tight bounds",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default="",
        help="1-based pages/ranges to crop (example: '1,3,5-8'). Default: all pages",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "boxes", "render"],
        default="auto",
        help="Cropping method: auto (render then boxes), boxes only, or render only",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=250,
        help="Pixel threshold [0-255] for considering content non-white in render mode",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pages = parse_page_list(args.pages) if args.pages else None
    crop_pdf(
        args.input,
        args.output,
        padding=args.padding,
        pages=pages,
        method=args.method,
        white_threshold=args.white_threshold,
    )


if __name__ == "__main__":
    main()
