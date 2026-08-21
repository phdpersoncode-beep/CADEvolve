"""Iteration over ``enumerate`` and ``zip`` rather than ``range``.

Stresses: tuple loop targets, and literal container iterables that the optional
unroll mode has to bind element-wise.
"""
import cadquery as cq

body_length = 60.0
body_width = 24.0
body_height = 14.0
slot_depths = [2.0, 3.5, 5.0]
slot_widths = [4.0, 5.0, 6.0]
slot_spacing = 14.0
end_hole_diameter = 5.0

body = cq.Workplane("XY").box(body_length, body_width, body_height)

for position, (depth, width) in enumerate(zip(slot_depths, slot_widths)):
    x_offset = -slot_spacing + position * slot_spacing
    body = (
        body.faces(">Z")
        .workplane()
        .center(x_offset, 0.0)
        .rect(width, body_width)
        .cutBlind(-depth)
    )

for side in (">X", "<X"):
    body = (
        body.faces(side)
        .workplane(centerOption="CenterOfMass")
        .circle(end_hole_diameter / 2)
        .cutBlind(-6.0)
    )

result = body
