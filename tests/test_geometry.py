"""
Unit tests for geometry.py.
Run: pytest tests/test_geometry.py -v
"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from geometry import (
    shoelace_area, sutherland_hodgman, polygon_iou,
    point_in_polygon, anchor_point, aabbs_overlap, bbox_to_polygon
)


# ─── shoelace_area ──────────────────────────────────────────────

class TestShoelaceArea:
    def test_unit_square(self):
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert shoelace_area(sq) == pytest.approx(1.0)

    def test_rectangle(self):
        rect = [(0, 0), (4, 0), (4, 3), (0, 3)]
        assert shoelace_area(rect) == pytest.approx(12.0)

    def test_triangle(self):
        tri = [(0, 0), (4, 0), (0, 3)]
        assert shoelace_area(tri) == pytest.approx(6.0)

    def test_clockwise_same_result(self):
        cw  = [(0, 0), (0, 1), (1, 1), (1, 0)]
        ccw = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert shoelace_area(cw) == pytest.approx(shoelace_area(ccw))

    def test_degenerate_line(self):
        assert shoelace_area([(0, 0), (1, 0)]) == 0.0

    def test_empty(self):
        assert shoelace_area([]) == 0.0


# ─── sutherland_hodgman ─────────────────────────────────────────

class TestSutherlandHodgman:
    def test_full_overlap(self):
        box  = [(0, 0), (2, 0), (2, 2), (0, 2)]
        clip = [(0, 0), (4, 0), (4, 4), (0, 4)]
        inter = sutherland_hodgman(box, clip)
        assert shoelace_area(inter) == pytest.approx(4.0, rel=1e-4)

    def test_partial_overlap(self):
        box  = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        clip = [(0, 0), (2, 0), (2, 2), (0, 2)]
        inter = sutherland_hodgman(box, clip)
        assert shoelace_area(inter) == pytest.approx(1.0, rel=1e-4)

    def test_no_overlap(self):
        box  = [(5, 5), (7, 5), (7, 7), (5, 7)]
        clip = [(0, 0), (2, 0), (2, 2), (0, 2)]
        inter = sutherland_hodgman(box, clip)
        assert len(inter) == 0 or shoelace_area(inter) == pytest.approx(0.0, abs=1e-6)


# ─── polygon_iou ────────────────────────────────────────────────

class TestPolygonIoU:
    def test_perfect_overlap(self):
        slot = [(0, 0), (10, 0), (10, 10), (0, 10)]
        area = shoelace_area(slot)
        iou  = polygon_iou((0, 0, 10, 10), slot, area)
        assert iou == pytest.approx(1.0, rel=1e-4)

    def test_no_overlap(self):
        slot = [(0, 0), (5, 0), (5, 5), (0, 5)]
        area = shoelace_area(slot)
        iou  = polygon_iou((10, 10, 15, 15), slot, area)
        assert iou == pytest.approx(0.0, abs=1e-6)

    def test_half_overlap(self):
        slot = [(0, 0), (10, 0), (10, 10), (0, 10)]
        area = shoelace_area(slot)
        # bbox covers left half
        iou = polygon_iou((0, 0, 5, 10), slot, area)
        # inter=50, union=100+50-50=100, iou=0.5
        assert iou == pytest.approx(0.5, rel=0.01)

    def test_zero_slot_area(self):
        assert polygon_iou((0, 0, 5, 5), [], 0.0) == 0.0


# ─── point_in_polygon ───────────────────────────────────────────

class TestPointInPolygon:
    SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]

    def test_centre_inside(self):
        assert point_in_polygon(5, 5, self.SQUARE) is True

    def test_outside(self):
        assert point_in_polygon(15, 5, self.SQUARE) is False
        assert point_in_polygon(-1, 5, self.SQUARE) is False

    def test_near_edge_inside(self):
        assert point_in_polygon(0.1, 5, self.SQUARE) is True

    def test_triangle(self):
        tri = [(0, 0), (10, 0), (5, 10)]
        assert point_in_polygon(5, 3,  tri) is True
        assert point_in_polygon(0, 9,  tri) is False
        assert point_in_polygon(5, 9,  tri) is True


# ─── aabbs_overlap ──────────────────────────────────────────────

class TestAABBsOverlap:
    def test_overlap(self):
        assert aabbs_overlap((0,0,5,5), (3,3,8,8)) is True

    def test_touching_edge(self):
        # Touching at x=5: depends on strict/non-strict; our impl is non-strict
        assert aabbs_overlap((0,0,5,5), (5,0,10,5)) is True

    def test_no_overlap(self):
        assert aabbs_overlap((0,0,3,3), (4,4,8,8)) is False

    def test_contained(self):
        assert aabbs_overlap((0,0,10,10), (2,2,4,4)) is True
