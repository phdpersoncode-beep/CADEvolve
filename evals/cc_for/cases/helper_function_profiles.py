"""A helper that returns a Workplane, called from inside a loop.

Stresses: chains inside a function body must be lowered too, and the call site in
the loop must become an explicit step without inlining the helper.
"""
import cadquery as cq

hub_radius = 9.0
hub_height = 10.0
blade_count = 5
blade_length = 22.0
blade_width = 3.0
blade_height = 8.0
blade_twist_deg = 12.0
bore_diameter = 6.0


def blade_profile(length, width, height, angle_deg):
    return (
        cq.Workplane("XY")
        .rect(width, length)
        .extrude(height)
        .translate((0.0, length / 2 + hub_radius * 0.6, 0.0))
        .rotate((0, 0, 0), (0, 0, 1), angle_deg)
    )


impeller = cq.Workplane("XY").circle(hub_radius).extrude(hub_height)

for blade_index in range(blade_count):
    angle = blade_index * (360.0 / blade_count) + blade_twist_deg
    blade = blade_profile(blade_length, blade_width, blade_height, angle)
    impeller = impeller.union(blade)

result = impeller.faces(">Z").workplane().circle(bore_diameter / 2).cutThruAll()
