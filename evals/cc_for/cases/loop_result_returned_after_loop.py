"""A helper that returns a name its loop assigned, where the loop may not run.

Stresses: the loop body binds ``solid`` and the ``return`` after the loop reads
it. Recording an alias for the body's assignment and exporting it past the loop
makes the return read a ``wpN`` that only exists if the loop ran -- and Python
does not promise that. With ``rib_count`` at zero the canonical program raises
``UnboundLocalError`` where the source returns the plate untouched.

The name has to keep a real binding across the loop instead, so the read after
it resolves exactly when the source's would.
"""
import cadquery as cq

plate_length = 60.0
plate_width = 40.0
plate_thickness = 6.0
rib_count = 0
rib_width = 4.0
rib_height = 5.0
rib_pitch = 12.0


def add_ribs(solid):
    face = solid.faces(">Z")
    for index in range(rib_count):
        solid = (
            face.workplane()
            .center(index * rib_pitch - rib_pitch, 0)
            .rect(rib_width, rib_width)
            .extrude(rib_height)
        )
    return solid


base = cq.Workplane("XY").box(plate_length, plate_width, plate_thickness)
result = add_ribs(base)
