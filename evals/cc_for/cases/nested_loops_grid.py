"""Nested loops building a point grid, with parameters reused in both loops.

Stresses: two nested ``for`` bodies whose targets (``row``, ``col``) and locals
(``x``, ``y``) are written once per iteration, plus one parameter (``pitch``) read
from both loop levels.
"""
import cadquery as cq

plate_x = 70.0
plate_y = 50.0
plate_z = 6.0
rows = 3
cols = 4
pitch = 12.0
hole_diameter = 4.0
boss_diameter = 7.0
boss_height = 2.0

centers = []
for row in range(rows):
    for col in range(cols):
        x = (col - (cols - 1) / 2) * pitch
        y = (row - (rows - 1) / 2) * pitch
        centers.append((x, y))

plate = cq.Workplane("XY").box(plate_x, plate_y, plate_z)
bossed = (
    plate.faces(">Z")
    .workplane()
    .pushPoints(centers)
    .circle(boss_diameter / 2)
    .extrude(boss_height)
)
result = (
    bossed.faces(">Z")
    .workplane()
    .pushPoints(centers)
    .hole(hole_diameter)
)
