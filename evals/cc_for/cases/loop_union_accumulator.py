"""Loop-carried geometry accumulator seeded with None.

Stresses: the accumulator must stay one stable name across iterations (SSA cannot
version it), and the ``None`` initializer must survive because the first iteration
reads it.
"""
import cadquery as cq

tooth_count = 6
tooth_width = 4.0
tooth_depth = 3.0
tooth_height = 5.0
pitch = 9.0
base_length = 60.0
base_width = 12.0
base_thickness = 6.0

rack = None
for index in range(tooth_count):
    offset_x = -base_length / 2 + pitch / 2 + index * pitch
    tooth = (
        cq.Workplane("XY")
        .box(tooth_width, tooth_depth, tooth_height)
        .translate((offset_x, 0.0, base_thickness / 2 + tooth_height / 2))
    )
    rack = tooth if rack is None else rack.union(tooth)

base = cq.Workplane("XY").box(base_length, base_width, base_thickness)
result = base.union(rack)
