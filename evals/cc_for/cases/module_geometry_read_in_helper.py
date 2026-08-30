"""A helper that reads a module-level geometry value, not one passed to it.

Stresses: lowering rewrites ``sweep_path`` to the ``wpN`` that carries its value
and drops the original assignment, which is correct for every module-level read.
A ``def`` resolves the name at call time against the module globals, where that
dropped assignment was the only binding -- so the alias has to be materialized
too, or the helper raises ``NameError`` the moment it is called.

The two Zero-to-CAD programs that hit this both build a swept profile once at
module level and consume it from inside a groove helper.
"""
import cadquery as cq

arc_radius = 60.0
span = 70.0
profile_width = 14.0
profile_height = 18.0
groove_width = 5.0
groove_height = 6.0
groove_offset = 3.0

sweep_path = (
    cq.Workplane("XY")
    .radiusArc((span, 0.0), arc_radius)
    .wire()
)

body = (
    cq.Workplane("XY")
    .transformed(rotate=(90, 0, 0))
    .rect(profile_width, profile_height)
    .sweep(sweep_path, transition="round")
)


def groove(offset):
    return (
        cq.Workplane("XY")
        .transformed(rotate=(90, 0, 0))
        .rect(groove_width, groove_height)
        .translate((-offset, 0, 0))
        .sweep(sweep_path, transition="round")
    )


result = body.cut(groove(groove_offset))
