"""An ``if``/``else`` that picks between two different modelling operations.

Stresses: both branches assign the same name, so versioning has to keep them
consistent at the join point instead of renaming one branch away.
"""
import cadquery as cq

use_rounded_corners = True
add_center_boss = False
plate_length = 55.0
plate_width = 35.0
plate_thickness = 7.0
corner_radius = 5.0
chamfer_size = 2.0
boss_diameter = 12.0
boss_height = 5.0
relief_diameter = 9.0

plate = cq.Workplane("XY").box(plate_length, plate_width, plate_thickness)

if use_rounded_corners:
    shaped = plate.edges("|Z").fillet(corner_radius)
else:
    shaped = plate.edges("|Z").chamfer(chamfer_size)

if add_center_boss:
    featured = shaped.faces(">Z").workplane().circle(boss_diameter / 2).extrude(boss_height)
else:
    featured = shaped.faces(">Z").workplane().circle(relief_diameter / 2).cutBlind(-2.0)

result = featured
