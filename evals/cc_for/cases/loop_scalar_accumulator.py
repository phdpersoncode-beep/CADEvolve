"""A loop-carried *scalar* that drives every subsequent geometry step.

Stresses: ``current_z`` and ``current_radius`` are read and rewritten each pass, so
versioning them per statement would break the running total.  A converter that
hoists them into the preamble as constants silently changes the part.
"""
import cadquery as cq

stage_count = 5
start_radius = 18.0
radius_step = 2.5
stage_height = 4.0
fillet_radius = 0.6

current_z = 0.0
current_radius = start_radius
body = cq.Workplane("XY").circle(start_radius).extrude(stage_height)
current_z = current_z + stage_height

for stage in range(1, stage_count):
    current_radius = current_radius - radius_step
    body = (
        body.faces(">Z")
        .workplane()
        .circle(current_radius)
        .extrude(stage_height)
    )
    current_z = current_z + stage_height

result = body.edges("%CIRCLE").fillet(fillet_radius)
