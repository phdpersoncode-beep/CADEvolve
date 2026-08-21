"""``continue`` and ``break`` inside a modelling loop.

Stresses: a loop that cannot be unrolled by evaluating the iterable alone, and
whose geometry effect depends on a predicate over the loop variable.
"""
import cadquery as cq

plate_side = 60.0
plate_thickness = 5.0
grid_pitch = 10.0
grid_count = 5
vent_diameter = 4.0
skip_modulus = 3
max_vents = 12

plate = cq.Workplane("XY").box(plate_side, plate_side, plate_thickness)

vent_points = []
placed = 0
for cell in range(grid_count * grid_count):
    if cell % skip_modulus == 0:
        continue
    if placed >= max_vents:
        break
    row = cell // grid_count
    col = cell % grid_count
    x = (col - (grid_count - 1) / 2) * grid_pitch
    y = (row - (grid_count - 1) / 2) * grid_pitch
    vent_points.append((x, y))
    placed = placed + 1

result = (
    plate.faces(">Z")
    .workplane()
    .pushPoints(vent_points)
    .hole(vent_diameter)
)
