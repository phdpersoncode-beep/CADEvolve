"""Revolve, sweep along a spline, and loft between wires.

Stresses: operations whose arguments are wires and paths rather than scalars, so
the lowering step has to keep intermediate geometry objects addressable.
"""
import cadquery as cq
import math

profile_radius = 12.0
profile_height = 20.0
wall = 2.0
revolve_angle = 300.0
path_points = 9
path_radius = 26.0
path_rise = 14.0
sweep_diameter = 4.0
loft_bottom = 16.0
loft_top = 8.0
loft_height = 10.0

shell_profile = (
    cq.Workplane("XZ")
    .moveTo(profile_radius - wall, 0.0)
    .lineTo(profile_radius, 0.0)
    .lineTo(profile_radius, profile_height)
    .lineTo(profile_radius - wall, profile_height)
    .close()
)
revolved = shell_profile.revolve(revolve_angle, (0, 0, 0), (0, 1, 0))

helix_points = [
    (
        path_radius * math.cos(2 * math.pi * i / (path_points - 1)),
        path_radius * math.sin(2 * math.pi * i / (path_points - 1)),
        path_rise * i / (path_points - 1),
    )
    for i in range(path_points)
]
sweep_path = cq.Workplane("XY").spline(helix_points, includeCurrent=False)
swept = (
    cq.Workplane("YZ")
    .center(helix_points[0][1], helix_points[0][2])
    .circle(sweep_diameter / 2)
    .sweep(sweep_path, isFrenet=True)
)

lofted = (
    cq.Workplane("XY")
    .workplane(offset=profile_height)
    .rect(loft_bottom, loft_bottom)
    .workplane(offset=loft_height)
    .circle(loft_top / 2)
    .loft(ruled=True)
)

result = revolved.union(swept).union(lofted)
