"""The ``cq.Sketch`` API rather than the Workplane 2-D API.

Stresses: a second fluent builder type whose chains also have to be lowered, and
whose ``.finalize()`` hands control back to a Workplane.
"""
import cadquery as cq

body_length = 50.0
body_width = 34.0
body_height = 12.0
pocket_length = 30.0
pocket_width = 18.0
pocket_fillet = 4.0
pocket_depth = 6.0
vent_radius = 2.5
vent_offset = 11.0

pocket_sketch = (
    cq.Sketch()
    .rect(pocket_length, pocket_width)
    .vertices()
    .fillet(pocket_fillet)
    .push([(-vent_offset, 0.0), (vent_offset, 0.0)])
    .circle(vent_radius, mode="s")
)

body = cq.Workplane("XY").box(body_length, body_width, body_height)
result = (
    body.faces(">Z")
    .workplane()
    .placeSketch(pocket_sketch)
    .cutBlind(-pocket_depth)
)
