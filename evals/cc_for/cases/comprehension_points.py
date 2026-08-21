"""List and nested comprehensions producing modelling coordinates.

Stresses: comprehensions have their own scope; renaming their targets as if they
were module-level names would corrupt them.
"""
import cadquery as cq
import math

disc_radius = 30.0
disc_thickness = 6.0
bolt_count = 8
bolt_circle_radius = 22.0
bolt_diameter = 4.0
slot_count = 4
slot_length = 8.0
slot_width = 3.0

bolt_points = [
    (
        bolt_circle_radius * math.cos(2 * math.pi * i / bolt_count),
        bolt_circle_radius * math.sin(2 * math.pi * i / bolt_count),
    )
    for i in range(bolt_count)
]
slot_angles = [360.0 * i / slot_count for i in range(slot_count)]
slot_points = [
    (
        (bolt_circle_radius * 0.45) * math.cos(math.radians(a)),
        (bolt_circle_radius * 0.45) * math.sin(math.radians(a)),
    )
    for a in slot_angles
]

disc = cq.Workplane("XY").circle(disc_radius).extrude(disc_thickness)
drilled = disc.faces(">Z").workplane().pushPoints(bolt_points).hole(bolt_diameter)
result = (
    drilled.faces(">Z")
    .workplane()
    .pushPoints(slot_points)
    .slot2D(slot_length, slot_width, 0)
    .cutThruAll()
)
