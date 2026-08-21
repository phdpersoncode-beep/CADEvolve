"""A name that is first a scalar parameter and later rebound to geometry.

Stresses: reaching-definition renaming has to keep the two meanings of ``body``
and ``height`` apart without letting a later geometry binding capture an earlier
numeric read.
"""
import cadquery as cq

height = 18.0
width = 40.0
body = height * 1.5
depth = body - 4.0

base = cq.Workplane("XY").box(width, depth, height)
body = base.faces(">Z").workplane().rect(width - 8.0, depth - 8.0).extrude(6.0)
height = body.faces(">Z").workplane().circle(5.0).cutBlind(-4.0)

width = 3.0
result = height.edges("|Z").fillet(width)
