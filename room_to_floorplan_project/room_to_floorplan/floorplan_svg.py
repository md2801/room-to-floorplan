from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from skimage import measure


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def remove_small_components(mask: np.ndarray, min_area_px: int = 64) -> np.ndarray:
    """Remove small connected components from a binary mask."""
    mask_u8 = _as_bool(mask).astype(np.uint8)
    if min_area_px <= 1:
        return mask_u8.astype(bool)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            out[labels == i] = 1
    return out.astype(bool)


def clean_mask(
    prob_or_mask: np.ndarray,
    threshold: float = 0.5,
    close_kernel: int = 5,
    open_kernel: int = 3,
    dilate_iterations: int = 0,
    erode_iterations: int = 0,
    min_area_px: int = 80,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Convert a probability map or binary mask into a cleaner binary mask.

    This is deliberately conservative: it removes speckles, fills tiny gaps,
    and preserves the broad shape predicted by the model.
    """
    arr = np.asarray(prob_or_mask)
    if arr.dtype == bool:
        mask = arr.astype(np.uint8)
    else:
        mask = (arr >= threshold).astype(np.uint8)

    if close_kernel > 1:
        k = np.ones((close_kernel, close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if open_kernel > 1:
        k = np.ones((open_kernel, open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    if dilate_iterations > 0:
        k = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, k, iterations=dilate_iterations)

    if erode_iterations > 0:
        k = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, k, iterations=erode_iterations)

    if fill_holes:
        # Flood-fill the background from the image border, then invert to get holes.
        flood = mask.copy()
        h, w = flood.shape[:2]
        flood_pad = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, flood_pad, (0, 0), 1)
        holes = (flood == 0).astype(np.uint8)
        mask = np.maximum(mask, holes)

    mask = remove_small_components(mask, min_area_px=min_area_px).astype(np.uint8)
    return mask.astype(bool)


def mask_to_polygons(
    mask: np.ndarray,
    min_area_px: float = 40.0,
    simplify_px: float = 1.5,
) -> List[Polygon]:
    """Vectorise a raster mask into simplified Shapely polygons."""
    mask_f = _as_bool(mask).astype(np.float32)
    contours = measure.find_contours(mask_f, 0.5)
    polygons: List[Polygon] = []

    for contour in contours:
        if len(contour) < 3:
            continue

        # skimage contours are row, col. SVG uses x, y.
        coords = [(float(col), float(row)) for row, col in contour]

        try:
            poly = Polygon(coords).buffer(0)
        except Exception:
            continue

        if poly.is_empty:
            continue

        geoms = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
        for geom in geoms:
            if geom.is_empty or geom.area < min_area_px:
                continue
            if simplify_px > 0:
                geom = geom.simplify(simplify_px, preserve_topology=True)
            if not geom.is_empty and geom.area >= min_area_px:
                polygons.append(geom)

    return polygons


def polygon_to_svg_path(poly: Polygon) -> str:
    """Convert a polygon, including holes, to an SVG path string."""
    def ring_to_d(coords: Iterable[Tuple[float, float]]) -> str:
        pts = list(coords)
        if not pts:
            return ""
        d = f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"
        for x, y in pts[1:]:
            d += f" L {x:.2f} {y:.2f}"
        return d + " Z"

    d = ring_to_d(poly.exterior.coords)
    for interior in poly.interiors:
        d += " " + ring_to_d(interior.coords)
    return d


def save_mask_png(mask: np.ndarray, out_path: str | Path) -> None:
    Image.fromarray((_as_bool(mask).astype(np.uint8) * 255)).save(out_path)


def save_probability_png(prob: np.ndarray, out_path: str | Path) -> None:
    arr = np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0)
    Image.fromarray((arr * 255).astype(np.uint8)).save(out_path)


def save_floorplan_overlay_png(
    structure_mask: np.ndarray,
    furniture_mask: Optional[np.ndarray],
    out_path: str | Path,
) -> None:
    """Save a simple RGB preview: black walls, grey furniture, white background."""
    structure = _as_bool(structure_mask)
    furniture = None if furniture_mask is None else _as_bool(furniture_mask)

    h, w = structure.shape
    canvas = np.full((h, w, 3), 255, np.uint8)

    if furniture is not None:
        canvas[furniture] = np.array([185, 185, 185], dtype=np.uint8)

    canvas[structure] = np.array([20, 20, 20], dtype=np.uint8)
    Image.fromarray(canvas).save(out_path)


def create_floorplan_svg_string(
    structure_mask: np.ndarray,
    furniture_mask: Optional[np.ndarray] = None,
    meters_per_pixel: Optional[float] = None,
    min_area_px: float = 40.0,
    simplify_px: float = 1.5,
    include_furniture: bool = True,
    title: str = "Predicted floorplan",
) -> str:
    """
    Convert predicted masks into a downloadable SVG floorplan.

    The output keeps image/grid coordinates in the SVG viewBox. If meters_per_pixel
    is available, it is stored as metadata so a downstream app can calculate scale.
    """
    structure = _as_bool(structure_mask)
    h, w = structure.shape

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(w),
            "height": str(h),
            "viewBox": f"0 0 {w} {h}",
            "role": "img",
            "aria-label": title,
        },
    )

    ET.SubElement(svg, "title").text = title

    metadata = ET.SubElement(svg, "metadata")
    metadata.text = (
        f"meters_per_pixel={meters_per_pixel:.8f}" if meters_per_pixel is not None else "meters_per_pixel=unknown"
    )

    ET.SubElement(svg, "rect", {"x": "0", "y": "0", "width": str(w), "height": str(h), "fill": "white"})

    if include_furniture and furniture_mask is not None:
        furniture_polys = mask_to_polygons(furniture_mask, min_area_px=min_area_px, simplify_px=simplify_px)
        furn_g = ET.SubElement(svg, "g", {"id": "furniture", "fill": "#d0d0d0", "stroke": "#8a8a8a", "stroke-width": "1"})
        for poly in furniture_polys:
            ET.SubElement(furn_g, "path", {"d": polygon_to_svg_path(poly)})

    structure_polys = mask_to_polygons(structure, min_area_px=min_area_px, simplify_px=simplify_px)

    # Merge overlapping structure polygons where possible to reduce SVG complexity.
    merged = unary_union(structure_polys) if structure_polys else []
    if isinstance(merged, Polygon):
        structure_polys = [merged]
    elif isinstance(merged, MultiPolygon):
        structure_polys = list(merged.geoms)

    wall_g = ET.SubElement(
        svg,
        "g",
        {
            "id": "walls",
            "fill": "#111111",
            "stroke": "none",
            "fill-rule": "evenodd",
        },
    )
    
    for poly in structure_polys:
        if poly.is_empty or poly.area < min_area_px:
            continue
    
        ET.SubElement(
            wall_g,
            "path",
            {
                "d": polygon_to_svg_path(poly),
                "fill": "#111111",
                "stroke": "none",
            },
        )

    return ET.tostring(svg, encoding="unicode")


def save_floorplan_svg(
    structure_mask: np.ndarray,
    furniture_mask: Optional[np.ndarray],
    out_svg: str | Path,
    meters_per_pixel: Optional[float] = None,
    min_area_px: float = 40.0,
    simplify_px: float = 1.5,
    include_furniture: bool = True,
    title: str = "Predicted floorplan",
) -> None:
    svg = create_floorplan_svg_string(
        structure_mask=structure_mask,
        furniture_mask=furniture_mask,
        meters_per_pixel=meters_per_pixel,
        min_area_px=min_area_px,
        simplify_px=simplify_px,
        include_furniture=include_furniture,
        title=title,
    )
    Path(out_svg).write_text(svg, encoding="utf-8")
