"""A bare statement that reads ``result`` before the terminal alias is bound.

Stresses: ``result`` is rebound inside a loop, so it becomes a loop-carried state
name.  Every read of it has to follow that rename -- including reads in
statements that assign nothing, such as the self-check assertion below, which is
a common idiom in generated CAD programs.
"""
import cadquery as cq

body_size = 24.0
bore_diameter = 5.0
edge_fillet = 1.5
expected_solids = 1

result = cq.Workplane("XY").box(body_size, body_size, body_size)

for selector in (">X", "<X", ">Y"):
    result = result.faces(selector).workplane().hole(bore_diameter)

assert result.solids().size() == expected_solids

result = result.edges("|Z").fillet(edge_fillet)
