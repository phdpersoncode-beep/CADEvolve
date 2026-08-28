"""A ``while`` loop, which is not a ``for`` loop and must never be unrolled.

Stresses: the converter's loop handling has to leave dynamic control flow alone
while still lowering the CadQuery chains inside its body.
"""
import cadquery as cq

base_radius = 20.0
min_radius = 6.0
taper_ratio = 0.78
layer_height = 3.0
bore_diameter = 5.0

radius = base_radius
tower = cq.Workplane("XY").circle(radius).extrude(layer_height)

while radius * taper_ratio >= min_radius:
    radius = radius * taper_ratio
    tower = tower.faces(">Z").workplane().circle(radius).extrude(layer_height)

result = tower.faces(">Z").workplane().circle(bore_diameter / 2).cutThruAll()
