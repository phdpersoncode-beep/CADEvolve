"""A result holding several disjoint solids that are never unioned.

Stresses: body counting.  ``Workplane.val()`` returns only the first solid, so a
converter (or a validator) that compares ``val()`` cannot see a dropped body.
"""
import cadquery as cq

pin_count = 4
pin_diameter = 5.0
pin_height = 18.0
pin_spacing = 14.0
cap_diameter = 8.0
cap_height = 2.5

pin_centers = [((i - (pin_count - 1) / 2) * pin_spacing, 0.0) for i in range(pin_count)]

pins = (
    cq.Workplane("XY")
    .pushPoints(pin_centers)
    .circle(pin_diameter / 2)
    .extrude(pin_height, combine=False)
)
caps = (
    cq.Workplane("XY")
    .workplane(offset=pin_height)
    .pushPoints(pin_centers)
    .circle(cap_diameter / 2)
    .extrude(cap_height, combine=False)
)

result = pins.union(caps, clean=True)
