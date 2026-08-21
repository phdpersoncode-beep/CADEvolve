"""One very deep uninterrupted fluent chain.

Stresses: chain lowering depth.  A single expression with twenty-plus chained calls
must become twenty-plus explicit ``wpN`` steps with no nesting left behind.
"""
import cadquery as cq

outline_x = 46.0
outline_y = 30.0
notch = 7.0
thickness = 6.0
hole_diameter = 4.5
hole_inset = 6.0
corner_fillet = 2.0
lip_height = 2.0

result = (
    cq.Workplane("XY")
    .moveTo(-outline_x / 2, -outline_y / 2)
    .lineTo(outline_x / 2 - notch, -outline_y / 2)
    .lineTo(outline_x / 2, -outline_y / 2 + notch)
    .lineTo(outline_x / 2, outline_y / 2)
    .lineTo(-outline_x / 2 + notch, outline_y / 2)
    .lineTo(-outline_x / 2, outline_y / 2 - notch)
    .close()
    .extrude(thickness)
    .edges("|Z")
    .fillet(corner_fillet)
    .faces(">Z")
    .workplane()
    .rect(outline_x - 2 * hole_inset, outline_y - 2 * hole_inset, forConstruction=True)
    .vertices()
    .hole(hole_diameter)
    .faces(">Z")
    .workplane()
    .rect(outline_x - 4.0, outline_y - 4.0)
    .extrude(lip_height)
    .faces(">Z")
    .workplane()
    .rect(outline_x - 8.0, outline_y - 8.0)
    .cutBlind(-lip_height)
)
