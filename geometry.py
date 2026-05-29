"""
Geometric primitives for parking slot detection.
All polygon vertices are (x, y) pixel coordinates.
"""
from __future__ import annotations
from typing import List, Tuple

Point = Tuple[float, float]
Polygon = List[Point]


def shoelace_area(polygon: Polygon) -> float:
    """Signed shoelace area; positive = counter-clockwise."""
    n = len(polygon)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def _left_of(ax: float, ay: float, bx: float, by: float,
              px: float, py: float) -> bool:
    """True if P is on the left side (or on) edge A→B (cross product ≥ 0)."""
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0


def _intersect(ax: float, ay: float, bx: float, by: float,
               cx: float, cy: float, dx: float, dy: float) -> Point:
    """Intersection of line AB with line CD."""
    a1 = by - ay
    b1 = ax - bx
    c1 = a1 * ax + b1 * ay

    a2 = dy - cy
    b2 = cx - dx
    c2 = a2 * cx + b2 * cy

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return ((ax + bx) / 2, (ay + by) / 2)
    return ((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det)


def sutherland_hodgman(subject: Polygon, clip: Polygon) -> Polygon:
    """
    Clip *subject* polygon against convex *clip* polygon.
    Returns the intersection polygon (possibly empty).
    """
    output = list(subject)
    if not output:
        return []

    n = len(clip)
    for i in range(n):
        if not output:
            return []
        inp = output
        output = []
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % n]

        for j in range(len(inp)):
            pcur = inp[j]
            pnxt = inp[(j + 1) % len(inp)]
            in_cur = _left_of(ax, ay, bx, by, pcur[0], pcur[1])
            in_nxt = _left_of(ax, ay, bx, by, pnxt[0], pnxt[1])

            if in_nxt:
                if not in_cur:
                    output.append(_intersect(ax, ay, bx, by,
                                             pcur[0], pcur[1], pnxt[0], pnxt[1]))
                output.append(pnxt)
            elif in_cur:
                output.append(_intersect(ax, ay, bx, by,
                                         pcur[0], pcur[1], pnxt[0], pnxt[1]))
    return output


def bbox_to_polygon(bbox: Tuple[float, float, float, float]) -> Polygon:
    """Convert (x1, y1, x2, y2) bbox to 4-vertex polygon (clockwise)."""
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def polygon_iou(bbox: Tuple[float, float, float, float],
                slot_polygon: Polygon,
                slot_area: float) -> float:
    """
    IoU between axis-aligned *bbox* and *slot_polygon* (convex).
    Uses Sutherland-Hodgman for intersection area.
    """
    x1, y1, x2, y2 = bbox
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if box_area == 0 or slot_area == 0:
        return 0.0

    box_poly = bbox_to_polygon(bbox)
    inter_poly = sutherland_hodgman(box_poly, slot_polygon)
    inter_area = shoelace_area(inter_poly) if len(inter_poly) >= 3 else 0.0

    union_area = box_area + slot_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def point_in_polygon(px: float, py: float, polygon: Polygon) -> bool:
    """
    Ray-casting even-odd rule.
    Each vertex counted at most once to avoid double-counting at vertices.
    """
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Strict one-sided test avoids double-count at shared vertex y
        if (yi > py) != (yj > py):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside
        j = i
    return inside


def anchor_point(bbox: Tuple[float, float, float, float]) -> Point:
    """Bottom-center of bbox — approximate ground contact of vehicle."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def aabb_of_polygon(polygon: Polygon) -> Tuple[float, float, float, float]:
    """Axis-aligned bounding box of a polygon: (xmin, ymin, xmax, ymax)."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def aabbs_overlap(a: Tuple[float, float, float, float],
                  b: Tuple[float, float, float, float]) -> bool:
    """Fast AABB overlap test."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
