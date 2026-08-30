"""A loop accumulator the loop *writes* without reading, updated under a branch.

Stresses: ``tallest`` is a named parameter by every syntactic test -- one simple
assignment of a float literal -- but the loop rebinds it, and only sometimes, so
reaching-definition renaming has to leave both definitions sharing the name.
Neither placement may then move the initializer across the loop.  CC-step would
sink it below (nothing in the loop *reads* ``tallest``, so the loop does not look
like a reader) and overwrite the accumulated height with 0.5; CC-for hoists it
upward, which is safe here only because the loop already follows it.

The program still builds and still passes the structural contract either way --
the boss is simply the wrong height -- so this case is a geometry gate, not an
AST one.
"""
import cadquery as cq

plate_size = 40.0
plate_thickness = 4.0
boss_diameter = 12.0
rib_heights = [1.0, 6.0, 2.0]
tall_threshold = 3.0

plate = cq.Workplane("XY").box(plate_size, plate_size, plate_thickness)

tallest = 0.5
for rib_height in rib_heights:
    if rib_height > tall_threshold:
        tallest = rib_height

result = (
    plate.faces(">Z")
    .workplane()
    .circle(boss_diameter / 2)
    .extrude(tallest)
)
