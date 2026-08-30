"""A namespace field left as ``None`` and computed once the others are known.

Stresses: flattening rewrites ``m.bolt.radius`` to a named parameter, which is
the whole point -- but this program writes the field back through the object
after construction, and the write and the read have to stay connected. Flatten
the read and it keeps seeing the ``None`` placeholder, so the modelling call gets
``None`` where it wanted a radius.

A field the program rebinds through the object is left reading through it.
"""
import cadquery as cq
from types import SimpleNamespace as Measures

m = Measures(
    outer_diameter=80.0,
    plate_thickness=6.0,
    rim_thickness=8.0,
    bolt=Measures(
        hole_diameter=5.0,
        count=4,
        start_angle=45.0,
        radius=None,
    ),
)
m.bolt.radius = 0.5 * m.outer_diameter - 0.5 * m.rim_thickness

result = (
    cq.Workplane("XY")
    .circle(0.5 * m.outer_diameter)
    .extrude(m.plate_thickness)
    .faces(">Z")
    .workplane()
    .polarArray(
        radius=m.bolt.radius,
        startAngle=m.bolt.start_angle,
        count=m.bolt.count,
        angle=360.0,
    )
    .hole(m.bolt.hole_diameter)
)
